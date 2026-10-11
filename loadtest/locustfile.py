"""Load test POST /translate.

  locust -f locustfile.py --host http://127.0.0.1:8000 --web-port 8090 \
      --src-lang en --trg-lang ar --batch-size 40 \
      --users 20 --spawn-rate 2 --num-requests 0

--users is how many Locust users run at once.
--spawn-rate is how many of those users start per second.
--num-requests stops the run after that many translate calls (0 keeps going).
"""

from locust import HttpUser, between, events, task
from locust.exception import StopTest

ENGLISH = [
    "Hello.",
    "How are you?",
    "The weather is nice today.",
    "Please open the window.",
    "I will arrive tomorrow morning.",
    "Where is the nearest station?",
    "This is a short test sentence.",
    "Thank you for your help.",
]

ARABIC = [
    "مرحبا.",
    "كيف حالك؟",
    "الجو جميل اليوم.",
    "من فضلك افتح النافذة.",
    "سأصل غدا صباحا.",
    "أين أقرب محطة؟",
    "هذه جملة اختبار قصيرة.",
    "شكرا لمساعدتك.",
]

_state = {"count": 0}


@events.init_command_line_parser.add_listener
def _add_arguments(parser):
    parser.add_argument("--src-lang", default="en", help="Source language code", include_in_web_ui=True)
    parser.add_argument("--trg-lang", default="ar", help="Target language code", include_in_web_ui=True)
    parser.add_argument("--batch-size", type=int, default=40, help="Number of texts in each request", include_in_web_ui=True)
    parser.add_argument("--num-requests", type=int, default=0, help="Stop after this many requests; 0 means until the run is stopped", include_in_web_ui=True)


@events.test_start.add_listener
def _on_test_start(environment, **kwargs):
    _state["count"] = 0
    options = environment.parsed_options
    if options.batch_size < 1:
        environment.runner.quit()
        raise RuntimeError("--batch-size must be at least 1")


def _batch(src_lang, size):
    pool = ARABIC if src_lang == "ar" else ENGLISH
    return [pool[index % len(pool)] for index in range(size)]


class TranslateUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def translate(self):
        options = self.environment.parsed_options
        self.client.post(
            "/translate",
            json={
                "texts": _batch(options.src_lang, options.batch_size),
                "src_lang": options.src_lang,
                "trg_lang": options.trg_lang,
            },
            timeout=360,
            name="/translate",
        )
        _state["count"] += 1
        if options.num_requests and _state["count"] >= options.num_requests:
            raise StopTest()
