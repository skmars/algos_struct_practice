######### "Python Cookbook" (David Beazley) ##########

# ----- Dictionaries ------#
most_spoken_languages = {
    "English": (1.452, "billion"),
    "Mandarin Chinese": (1.118, "billion"),
    "Hindi": (602.2, "million"),
    "Spanish": (548.3, "million"),
    "French": (274.1, "million"),
    "Modern Standard Arabic": (274.0, "million"),
    "Bengali": (272.7, "million"),
    "Russian": (258.2, "million"),
    "Portuguese": (257.7, "million"),
    "Urdu": (231.3, "million"),
    "Indonesian": (199.0, "million"),
    "Standard German": (134.6, "million"),
    "Japanese": (125.4, "million"),
    "Nigerian Pidgin": (120.7, "million"),
    "Marathi": (99.1, "million"),
    "Telugu": (95.7, "million"),
    "Turkish": (88.1, "million"),
    "Tamil": (86.4, "million"),
    "Yue Chinese": (85.6, "million"),
    "Vietnamese": (85.3, "million"),
    "Tagalog": (82.3, "million"),
}
countries_and_languages = [
    {
        "country": "Afghanistan",
        "first_official_lang": "Afgan-Persian",
        "second_lang": "Pashto",
    },
    {"country": "Canada", "first_official_lang": "English", "second_lang": "French"},
    {"country": "Algeria", "first_official_lang": "Arabic", "second_lang": "French"},
    {
        "country": "Iceland",
        "first_official_lang": "Icelandic",
        "second_lang": "English",
    },
    {
        "country": "Angola",
        "first_official_lang": "Portuguese",
        "second_lang": "Umbundu",
    },
    {
        "country": "Switzerland",
        "first_official_lang": "German",
        "second_lang": "French",
    },
]

## Sorting by common key ( itemgetter / attrgetter)
sorted_by_second_lang_with_lambda = sorted(
    countries_and_languages, key=lambda s: s["second_lang"]
)
sorted_by_second_lang_with_lambda_mult_keys = sorted(
    countries_and_languages, key=lambda s: (s["second_lang"], s["country"])
)

from operator import itemgetter

sorted_by_second_lang_with_itemgetter = sorted(
    countries_and_languages, key=itemgetter("second_lang")
)
sorted_by_second_lang_with_itemgetter_mult_keys = sorted(
    countries_and_languages, key=itemgetter("second_lang", "country")
)
# [
# {'country': 'Iceland', 'first_official_lang': 'Icelandic', 'second_lang': 'English'},
# {'country': 'Canada', 'first_official_lang': 'English', 'second_lang': 'French'},
# {'country': 'Algeria', 'first_official_lang': 'Arabic', 'second_lang': 'French'},
# {'country': 'Switzerland', 'first_official_lang': 'German', 'second_lang': 'French'},
# {'country': 'Afghanistan', 'first_official_lang': 'Afgan-Persian', 'second_lang': 'Pashto'},
# {'country': 'Angola', 'first_official_lang': 'Portuguese', 'second_lang': 'Umbundu'}
# ]

min(countries_and_languages, key=(itemgetter("country")))
# {'country': 'Afghanistan', 'first_official_lang': 'Afgan-Persian', 'second_lang': 'Pashto'}


# ----- Sequence ------#
words = [
    "Well",
    "I",
    "wish",
    "I",
    "could",
    "be",
    "like",
    "a",
    "bird",
    "in",
    "the",
    "sky",
]
seq_instead_dict = ["Eath", 4.543, "billion", 8.03, "milliard", (14, 4, 7531)]

## Unpacking
name, years, living_scale, population, pop_scale, Jesus_birth = seq_instead_dict
(
    name,
    years,
    _,
    population,
    _,
    _,
) = seq_instead_dict
mars_data = ("Mars", 4.603, "billion", "not_inhabitant", "UFO")
name, years, _, *population = mars_data


## Most Frequent in Sequence
from collections import Counter

words_counter = Counter(words)
words_counter.most_common(2)
# [('I', 2), ('Well', 1)]


# ----- Classes ------#
from random import randrange


class User:
    def __init__(self):
        self.user_id = randrange(11, 66)

    def __repr__(self):
        return f"User with id{self.user_id}"


## Sorting
diff_users = [User(), User(), User()]
sorted_with_lambda = sorted(diff_users, key=lambda u: u.user_id)

from operator import attrgetter

sorted_with_attrgetter = sorted(diff_users, key=attrgetter("user_id"))
