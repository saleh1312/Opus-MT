#!/usr/bin/env python

import os
import sys
import time
import json
import argparse
import websocket
from tornado import web, ioloop, queues, gen, process
from content_processor import ContentProcessor

class TranslatorInterface():
    """An interface to a single, possibly multilingual, model."""

    def __init__(self, srclang, targetlang, service, model):
        self.service = service
        self.contentprocessor = ContentProcessor(
            srclang,
            targetlang,
            sourcebpe=self.service.get('sourcebpe'),
            targetbpe=self.service.get('targetbpe'),
            sourcespm=self.service.get('sourcespm'),
            targetspm=self.service.get('targetspm')
        )
        self.worker = model
        # becomes nonempty if there are multiple target languages
        self.preamble = ""

    def translate(self, text):
        sentences = self.contentprocessor.preprocess(text)
        translatedSentences = self.worker.translate(self.preamble + '\n'.join(sentences))
        translation = self.contentprocessor.postprocess(translatedSentences)
        return ' '.join(translation)

    def translate_batch(self, texts):
        """Translate each text in one Marian call and return one string per input."""
        sentence_groups = []
        flat = []
        for text in texts:
            sentences = self.contentprocessor.preprocess(text)
            sentence_groups.append(sentences)
            flat.extend(sentences)
        if not flat:
            return ["" for _ in texts]

        # postprocess reads token counts from the full sentence list
        self.contentprocessor.sentences = list(flat)
        lines = []
        for sentence in flat:
            if self.preamble:
                lines.append(self.preamble + sentence)
            else:
                lines.append(sentence)
        translated = self.worker.translate('\n'.join(lines))
        if translated and translated[-1] == '' and len(translated) == len(flat) + 1:
            translated = translated[:-1]
        if len(translated) != len(flat):
            raise RuntimeError(
                "marian returned {} lines for {} sentences".format(len(translated), len(flat)))
        postprocessed = self.contentprocessor.postprocess(translated)
        results = []
        offset = 0
        for sentences in sentence_groups:
            count = len(sentences)
            results.append(' '.join(postprocessed[offset:offset + count]))
            offset += count
        return results

    def ready(self):
        return self.worker != None and self.worker.ready()

    def is_loaded(self):
        return self.worker != None and self.worker.is_loaded()

    def memory_bytes(self):
        if self.worker is None:
            return None
        return self.worker.memory_bytes()

    def on_exit(self):
        if self.worker != None:
            self.worker.on_exit()

# How long to wait for marian-server to load a model and open its port.
MODEL_LOAD_TIMEOUT = 300

_subprocess_initialized = False


def _ensure_subprocess_initialized():
    global _subprocess_initialized
    if not _subprocess_initialized:
        process.Subprocess.initialize()
        _subprocess_initialized = True


class TranslatorWorker():
    """One translation model, loaded on the first request that needs it.

    Uses a converted CTranslate2 model when models/<pair>/ct2 exists.
    Otherwise starts marian-server.
    """

    def __init__(self, host, port, configuration):
        self.host = host
        self.port = port
        self.configuration = configuration
        self.model_dir = os.path.dirname(configuration)
        self.ct2_dir = os.path.join(self.model_dir, "ct2")
        self.ws_url = "ws://{}:{}/translate".format(host, port)
        self.p = None
        self.translator = None
        self._memory_bytes = None

    def _ct2_ready(self):
        if not os.path.isfile(os.path.join(self.ct2_dir, "model.bin")):
            return False
        try:
            import ctranslate2  # noqa: F401
        except ImportError:
            return False
        return True

    def ensure_started(self):
        """Load this model if needed, and block until it can translate."""
        if self._ct2_ready():
            if self.translator is None:
                import ctranslate2
                print("Loading model {} ".format(self.ct2_dir), flush=True)
                before = _vm_rss_bytes()
                self.translator = ctranslate2.Translator(
                    self.ct2_dir, device="cpu", inter_threads=1, intra_threads=4)
                after = _vm_rss_bytes()
                if before is not None and after is not None:
                    self._memory_bytes = max(0, after - before)
                print("Model ready {}".format(self.ct2_dir), flush=True)
            return
        if self.p is not None and self.p.proc.poll() is None:
            if not self._accepting():
                self._wait_until_ready()
            return
        self._launch()
        self._wait_until_ready()

    def _launch(self):
        _ensure_subprocess_initialized()
        print("Starting marian-server for {} on port {}".format(
            self.configuration, self.port), flush=True)
        self.p = process.Subprocess(['marian-server', '-c',
                                     self.configuration,
                                     '-p', str(self.port),
                                     '--allow-unk',
                                     # enables translation with a mini-batch size of 64, i.e. translating 64 sentences at once, with a beam-size of 6.
                                     '-b', '6',
                                     '--mini-batch', '64',
                                     # use a length-normalization weight of 0.6 (this usually increases BLEU a bit).
                                     '--normalize', '0.6',
                                     '--maxi-batch-sort', 'src',
                                     '--maxi-batch', '100',
                                      ])
        self.p.set_exit_callback(self.on_exit)

    def _wait_until_ready(self):
        deadline = time.time() + MODEL_LOAD_TIMEOUT
        while time.time() < deadline:
            if self.p is None or self.p.proc.poll() is not None:
                raise RuntimeError(
                    "marian-server on port {} exited before it was ready".format(self.port))
            if self._accepting():
                print("marian-server ready on port {}".format(self.port), flush=True)
                return
            time.sleep(0.5)
        raise RuntimeError(
            "marian-server on port {} did not become ready within {}s".format(
                self.port, MODEL_LOAD_TIMEOUT))

    def on_exit(self, returncode=None):
        print("marian-server on port {} exited with code {}".format(
            self.port, returncode), flush=True)
        self.p = None

    def translate(self, sentences):
        self.ensure_started()
        if self.translator is not None:
            return self._translate_ct2(sentences)
        ws = websocket.create_connection(self.ws_url)
        ws.send(sentences)
        translatedSentences = ws.recv().split('\n')
        ws.close()
        return translatedSentences

    def _translate_ct2(self, sentences):
        lines = sentences.split('\n')
        tokens = [line.split() for line in lines]
        pending = [(index, token_row) for index, token_row in enumerate(tokens) if token_row]
        translated = [""] * len(lines)
        if not pending:
            return translated
        results = self.translator.translate_batch(
            [token_row for _, token_row in pending],
            beam_size=4,
            max_batch_size=64,
        )
        for (index, _), result in zip(pending, results):
            pieces = [token for token in result.hypotheses[0] if token not in ("</s>", "<s>")]
            translated[index] = " ".join(pieces)
        return translated

    def _accepting(self):
        if self.translator is not None:
            return True
        try:
            ws = websocket.create_connection(self.ws_url, timeout=1)
            ws.close()
        except Exception:
            return False
        return True

    def ready(self):
        # Not started yet: the HTTP server is up and this model loads on demand.
        if self.translator is None and self.p is None:
            return True
        return self._accepting()

    def is_loaded(self):
        if self.translator is not None:
            return True
        if self.p is None or self.p.proc.poll() is not None:
            return False
        return self._accepting()

    def memory_bytes(self):
        """Resident RAM for this model. None until it has been loaded."""
        if self.translator is not None:
            return self._memory_bytes
        if self.p is not None and self.p.proc.poll() is None:
            return _vm_rss_bytes(self.p.proc.pid)
        return None


def _vm_rss_bytes(pid=None):
    path = "/proc/{}/status".format(pid or os.getpid())
    try:
        with open(path) as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None

class ApiHandler(web.RequestHandler):
    def initialize(self, api, config, worker_pool):
        self.worker_pool = worker_pool
        self.config = config
        self.api = api
        self.worker = None
        self.args = {}

    def prepare_args(self):
        if self.request.headers['Content-Type'] == 'application/json':
            self.args = json.loads(self.request.body)

    def get(self):
        if self.api == 'ready':
            if all(map(lambda x: x.ready(), self.worker_pool.values())):
                self.set_status(204)
            else:
                self.set_status(500, "Translation server(s) not responding")
        elif self.api == 'languages':
            languages = {}
            for source_lang in self.config:
                languages[source_lang] = []
                targetLangs = self.config[source_lang]
                for target_lang in targetLangs:
                    languages[source_lang].append(target_lang)

            return self.write(dict(languages=languages))
        elif self.api == 'models':
            models = []
            for source_lang in self.config:
                targetLangs = self.config[source_lang]
                for target_lang in targetLangs:
                    lang_pair = "{}-{}".format(source_lang, target_lang)
                    pair_config = targetLangs[target_lang]
                    worker = self.worker_pool[lang_pair]
                    memory_bytes = worker.memory_bytes()
                    models.append({
                        "src_lang": source_lang,
                        "trg_lang": target_lang,
                        "loaded": worker.is_loaded(),
                        "memory_bytes": memory_bytes,
                        "memory_mb": round(memory_bytes / (1024.0 * 1024.0), 1) if memory_bytes is not None else None,
                        "configuration": pair_config.get("configuration"),
                        "port": pair_config.get("port"),
                    })
            loaded = [model for model in models if model["loaded"]]
            return self.write(dict(loaded=loaded, models=models))

    def post(self):
        self.prepare_args()
        if 'from' not in self.args or 'to' not in self.args:
            self.set_status(400)
            self.write(dict(error="from and to are required"))
            return
        lang_pair = "{}-{}".format(self.args['from'], self.args['to'])
        if lang_pair not in self.worker_pool:
            self.set_status(400)
            self.write(
                dict(error="Language pair {} not suppported".format(lang_pair)))
            return
        self.worker = self.worker_pool[lang_pair]
        try:
            if 'texts' in self.args:
                texts = self.args['texts']
                if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                    self.set_status(400)
                    self.write(dict(error="texts must be a list of strings"))
                    return
                translations = self.worker.translate_batch(texts)
                self.write(dict(translations=translations))
                return
            if 'source' not in self.args:
                self.set_status(400)
                self.write(dict(error="source or texts is required"))
                return
            translation = self.worker.translate(self.args['source'])
        except Exception as exc:
            self.set_status(503)
            self.write(dict(error="Failed to translate: {}".format(exc)))
            return
        self.write(dict(translation=translation))


class MainHandler(web.RequestHandler):
    def initialize(self, config):
        self.config = config

    def get(self):
        self.render("index.template.html", title="Opus MT")


def initialize_workers(config):
    worker_pool = {}
    models = {}
    for source_lang in config:
        targetLangs = config[source_lang]
        for target_lang in targetLangs:
            lang_pair = "{}-{}".format(source_lang, target_lang)
            pair_config = targetLangs[target_lang]
            if pair_config['configuration'] not in models:
                models[pair_config['configuration']] = TranslatorWorker(
                    pair_config['host'], pair_config['port'], pair_config['configuration'])
            worker_pool[lang_pair] = TranslatorInterface(
                source_lang, target_lang, pair_config, models[pair_config['configuration']])
            # Only multilingual models need a target-language token.
            # Dedicated pairs must not get one just because another pair shares the source.
            if pair_config.get('preamble'):
                worker_pool[lang_pair].preamble = pair_config['preamble']

    return worker_pool


settings = dict(
    template_path=os.path.join(os.path.dirname(__file__), "static"),
    static_path=os.path.join(os.path.dirname(__file__), "static"),
)


def make_app(args):
    services = {}
    with open(args.config, 'r') as configfile:
        services = json.load(configfile)
    worker_pool = initialize_workers(services)
    handlers = [
        (r"/", MainHandler, dict(config=services)),
        (r"/api/translate", ApiHandler,
         dict(api='translate', config=services, worker_pool=worker_pool)),
        (r"/api/ready", ApiHandler,
         dict(api='ready', config=services, worker_pool=worker_pool)),
        (r"/api/languages", ApiHandler,
         dict(api='languages', config=services, worker_pool=worker_pool)),
        (r"/api/models", ApiHandler,
         dict(api='models', config=services, worker_pool=worker_pool))
    ]
    application = web.Application(handlers, debug=False, **settings)
    return application


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Marian MT translation server.')
    parser.add_argument('-p', '--port', type=int, default=8888,
                        help='Port the server will listen on')
    parser.add_argument('-c', '--config', type=str, default="services.json",
                        help='MT server configurations')
    args = parser.parse_args()
    application = make_app(args)
    application.listen(args.port)
    ioloop.IOLoop.current().start()
