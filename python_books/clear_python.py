from abc import ABC, abstractmethod
import itertools
import random
import webbrowser
from typing import Union


# ######## Practice to "Clear Python" (Dan Bader) #########


class Validator(ABC):
    def __set_name__(self, owner, name):
        self.private_name = "_" + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        pass


class AccepatableMood(Validator):
    def __init__(self, *mood_options) -> None:
        self.options = set(mood_options)

    def validate(self, value):
        if value not in self.options:
            raise AttributeError(f"Attribute {value} must be one of {self.options}")

    def show_my_mood(self) -> str:
        print(
            f" All mood that  you have in your life: {self.options}\n"
            "And I will do whatever it takes!"
        )


class Yes:
    def __init__(self):
        self.yes = True

    def __call__(self):
        return self.yes


class No:
    def __init__(self):
        self.no = False

    def __call__(self):
        return self.no


class MoodAfterTechInterviews(Validator):
    need_to_fullfeel = True
    __slots__ = [
        "expected",
        "reality",
        "preinterview_passed",
    ]

    def __init__(self, expected="Happy being hired!", in_reality=None):
        self.expected = expected
        self.reality = in_reality
        self.preinterview_passed = True

    def __repr__(self, readyToTalk: Union[Yes, No]):
        if readyToTalk:
            print("If one door closes, another opens. Always remember...")

    def validate(self, value):
        "If it's possible to talk"
        if not isinstance(value, str):
            raise AttributeError("Say something to me, please!")
        assert "depression" not in value, "It could be done something! God is here!"

    @property
    def stress_level(self):
        "Stress level while interview"
        _user_stress_rate = input("How will you rate your stress level? (0 - 10: )")
        try:
            _user_stress_rate = int(_user_stress_rate)
        except:
            raise AttributeError("Use numbers (0 - 10) to rate: ")
        assert isinstance(_user_stress_rate, int), "Use numbers (0 - 10) to rate: "
        if not 0 < _user_stress_rate < 10:
            raise AttributeError("The stress is out of my mind")
        return _user_stress_rate

    @staticmethod
    def possible_help():
        fun_activities = [
            "Candy Crash",
            "Sunflower seeds",
            "Go for a walk",
        ]
        random.shuffle(fun_activities)
        print(f"There is a suggestion for you: {fun_activities[0]}")

    @classmethod
    def check_if_tech_interview(cls, try_=0):
        if try_ > 1:
            return

        possible_answers = (
            (
                "yes",
                "Y",
                "y",
                "Yes",
                "YES",
            ),
            (
                "NO",
                "No",
                "no",
                "N",
                "n",
            ),
        )

        user_input = input("Was that the tech interview?  Yes/No ")
        if user_input not in possible_answers[0]:
            cls.check_if_tech_interview()
            cls.need_to_fullfeel = False

    def days_to_recover(
        self,
    ):
        "How much hours you need to recover"
        if self.stress_level:
            print(f"You need about {random.randrange(0, 200)} days to recover")

    def speak_with_me(text, volume):
        def whisper():
            return text.lower() + "..."

        def yell():
            return text.upper() + "!"

        if volume > 0.5:
            return yell
        else:
            return whisper


class BeCold:
    "For contextmanager"

    def __init__(self):
        self.calm_down = "https://www.youtube.com/watch?v=nsDm36osJW8&t=1809s"
        webbrowser.register(
            "mozilla",
            None,
            webbrowser.BackgroundBrowser(
                "C:/Program Files/Mozilla Firefox/firefox.exe"
            ),
        )

    def __enter__(self):
        print("I am here. You are not alone. Tell me everyrhing")

    def __exit__(self, exc_type, exc_val, exc_tb):
        webbrowser.get(using="mozilla").open_new_tab(url=self.calm_down)
        # webbrowser.get(using="firefox").open_new(
        #     self.calm_down,
        # )


test_start = 1
with BeCold() as relax:
    print(MoodAfterTechInterviews(), Yes())
    MoodAfterTechInterviews.check_if_tech_interview()
    test_in = 2
    my_mood_today = MoodAfterTechInterviews(in_reality="Here we go a knife again.")
    my_mood_today.validate("I will survive")
    my_mood_today.possible_help()
    my_mood_today.days_to_recover()
test_end = 3

SECRET = "shhh!!!"


class Error:
    def __init__(self):
        pass


err = Error()
user_input = "{error.__init__.__globals__[SECRET]}"
print(user_input.format(error=err))
