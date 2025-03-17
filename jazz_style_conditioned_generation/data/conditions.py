#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Creates mappings for all conditions (performer, subgenre, mood, etc.)"""

import os

import numpy as np
from miditok import MusicTokenizer

from jazz_style_conditioned_generation import utils

MAX_GENRE_TOKENS_PER_TRACK = 5  # This is the maximum number of genre tokens we'll consider per track
MAX_SIMILAR_PIANISTS = 5  # This is the maximum number of "similar pianists" we'll get per track

# These are the only conditions we'll accept values for
ACCEPT_CONDITIONS = ["moods", "genres", "pianist", "themes"]
# Each list should be populated with values for a condition that we don't want to use
# TODO: now we've defined "INCLUDE", maybe we can remove this?
EXCLUDE = {
    # There are 117 "raw" genres scraped from TiVo
    "genres": [
        # Nearly every track could be described as one of these genres
        "Jazz Instrument",
        "Jazz",
        "Piano Jazz",
        "Keyboard",
        "Solo Instrumental",
        "Improvisation",
        # These tags seem incorrect given the type of music we know to be in the dataset
        "Big Band",
        "Choral",
        "Electronic",
        "Electro",
        "Club/Dance",
        "M-Base",  # Steve Coleman says that "M-Base is not a musical style"
        "Guitar Jazz",
        "Modern Big Band",
        "Orchestral",
        "Saxophone Jazz",
        "Symphony",
        "Trumpet Jazz",
        "Vibraphone/Marimba Jazz",
        "Vocal",
        "Vocal Music",
        "Vocal Pop",
    ],
    "moods": [],
    "pianist": [
        "JJA Pianist 1",
        "JJA Pianist 2",
        "JJA Pianist 3",
        "JJA Pianist 4",
        "JJA Pianist 5",
        "Doug McKenzie",
        # TODO: consider adding pianists who have fewer than N tracks here
        # These pianists all have fewer than 50 tracks
        'Roland Hanna',
        'Herbie Nichols',
        'Erroll Garner',
        'George Shearing',
        'Hampton Hawes',
        'Duke Jordan',
        'Andy Laverne',
        'Bill Cunliffe',
        'Stanley Cowell',
        'Alan Pasqua',
        'Barry Harris',
        'Alan Broadbent',
        'Bobby Timmons',
        'Jessica Williams',
        'Ellis Marsalis',
        'Denny Zeitlin',
        'Wynton Kelly',
        'Marcus Roberts',
        'George Cables',
        'Joanne Brackeen',
        'Beegie Adair',
        'Gonzalo Rubalcaba',
        'Simon Mulligan',
        'Horace Parlan',
        'David Berkman',
        'Red Garland',
        'George Colligan',
        'Paul Bley',
        'Don Grusin',
        'Eldar Djangirov',
        'Eric Reed',
        'Gene Harris',
        'Monty Alexander',
        'Randy Weston',
        'Lennie Tristano',
        'Jed Distler',
        'Clare Fischer',
        'Geri Allen',
        'Michel Camilo',
        'Adam Birnbaum',
        'Michel Legrand',
        'Marian McPartland',
        'Makoto Ozone',
        'John Colianni',
        'James Williams',
        'Donald Brown',
        'Buddy Montgomery',
        'Tigran Hamasyan',
        'Andr Previn',
        'Benny Green',
        'Geoffrey Keezer',
        'Ralph Sutton',
        'Mitchel Forman',
        'Larry Goldings',
        'Cyrus Chestnut',
        'Sonny Clark',
        'Bill Mays',
        'Allen Farnham',
        'Jacky Terrasson',
        'Adam Makowicz',
        'Hiromi',
        'Kirk Lightsey',
        'Ted Rosenthal',
        'Ellis Larkins',
        'Lynne Arriale',
        'Vijay Iyer',
        'Harold Mabern',
        'Phineas Newborn Jr',
        'Kenny Drew Jr',
        'Mike Wofford',
        'Toshiko Akiyoshi',
        'Richard Beirach',
        'John Taylor',
        'Jim McNeely',
        'Michel Petrucciani',
        'Earl Hines',
        'Don Friedman',
        'Gerald Clayton',
        'Hal Galper',
        'Walter Norris',
        'Marc Copland',
        'Mulgrew Miller',
        'Kris Davis',
        'Roger Kellaway',
        'John Campbell',
        'Jaki Byard',
        'Kenny Werner',
        'Jason Moran',
        'Steve Kuhn',
        'Mary Lou Williams',
        'Chris Anderson',
        'Eliane Elias',
        'Bill Charlap',
        'Edward Simon',
        'Robi Botos',
        'Joe Sample',
        'Paul Smith',
        'Dado Moroni',
        'Justin Kauflin',
        'Ramsey Lewis',
        'Lance Anderson',
        'Renee Rosnes',
        'Joey Alexander',
        'Ethan Iverson',
    ],
    "themes": [],
}
# Xu et al. (2023): 1.5 million musescore MIDI files, yet only 20 genre tags.
# Sarmento et al. (2023): remove genre tags with fewer than 100 appearances in a dataset of 25000 tracks
MERGE = {
    "genres": {
        "African": [
            "African Jazz",
            "African Folk",
            "African Traditions",
            "Township Jazz",
            "South African Folk",
            "Southern African",
            "Highlife"
        ],
        "Avant-Garde Jazz": [
            "Modern Free",
            "Free Improvisation",
            "Free Jazz",
            "Avant-Garde Jazz",
            "Progressive Jazz",
            "Modern Creative",
            "Modern Jazz",
            "Avant-Garde"
        ],
        "Blues": [
            "Blues",
            "Jazz Blues",
            "Boogie-Woogie"
        ],
        "Bop": [
            "Bop",
            "Bebop"
        ],
        "Caribbean": [
            "Calypso",
            "Cuban Jazz",
            "Afro-Cuban Jazz",
            "Caribbean Traditions"
        ],
        "Classical": [
            "Classical",
            "Chamber Jazz",
            "Classical Crossover",
            "Chamber Music",
            "Concerto",
            "Third Stream",
            "Modern Composition",
            "French",
            "Western European Traditions",
            "European Folk"
        ],
        "Cool Jazz": [
            "Cool",
            "West Coast Jazz"
        ],
        "Easy Listening": [
            "Piano/Easy Listening",
            "New Age",
            "Smooth Jazz",
            "Lounge",
            "Easy Listening"
        ],
        "Fusion": [
            "Funk",
            "Jazz Funk",
            "Fusion",
            "Jazz-Funk"
        ],
        "Global": [
            "Global Jazz",
            "International",
            "Central/West Asian Traditions"
        ],
        "Hard Bop": [
            "Hard Bop"
        ],
        "Latin": [
            "Latin",
            "Latin Jazz",
            "Venezuelan",  # i.e., "Latin America"
            "South American Traditions",
            "Brazilian",
            "Brazilian Pop",
            "Brazilian Jazz",
            "Brazilian Traditions"
        ],
        "Straight-Ahead Jazz": [
            "Mainstream Jazz",
            "Standards",
            "Straight-Ahead Jazz",
            "Contemporary Jazz",
            "Crossover Jazz"
        ],
        "Modal Jazz": [
            "Modal Music",
            "Modal Jazz"
        ],
        "Religious": [
            "Black Gospel",
            "Gospel",
            "Religious",
            "Holidays",
            "Christmas",
            "Holiday",
            "Spirituals"
        ],
        "Stage & Screen": [
            "Ballet",
            "Film Music",
            "Original Score",
            "Cast Recordings",
            "Show Tunes",
            "Film Score",
            "Spy Music",
            "Soundtracks",
            "Stage & Screen",
            "Show/Musical",
            "Musical Theater"
        ],
        "Soul Jazz": [
            "Soul Jazz"
        ],
        "Pop/Rock": [
            "Traditional Pop",
            "Adult Alternative Pop/Rock",
            "Alternative/Indie Rock",
            "American Popular Song",
            "Jazz-Pop",
            "Pop/Rock"
        ],
        "Post-Bop": [
            "Post-Bop",
            "Neo-Bop"
        ],
        "Traditional & Early Jazz": [
            "Trad Jazz",
            "Ragtime",
            "Early Jazz",
            "Swing",
            "Stride",
            "New Orleans Jazz",
            "New Orleans Jazz Revival",
            "Dixieland"
        ],
    },
    "moods": {},
    "pianist": {},
    "themes": {},
}
INCLUDE = {
    "genres": [
        'African',
        'Avant-Garde Jazz',
        'Blues',
        'Bop',
        'Caribbean',
        'Classical',
        'Cool Jazz',
        'Easy Listening',
        'Fusion',
        'Global',
        'Hard Bop',
        'Latin',
        'Modal Jazz',
        'Pop/Rock',
        'Post-Bop',
        'Religious',
        'Soul Jazz',
        'Stage & Screen',
        'Straight-Ahead Jazz',
        'Traditional & Early Jazz'
    ],
    "pianist": [
        'Abdullah Ibrahim',
        'Ahmad Jamal',
        'Art Tatum',
        'Bill Evans',
        'Brad Mehldau',
        'Bud Powell',
        'Cedar Walton',
        'Chick Corea',
        'Dave McKenna',
        'Dick Hyman',
        'Fred Hersch',
        'Hank Jones',
        'Herbie Hancock',
        'John Hicks',
        'Junior Mance',
        'Keith Jarrett',
        'Kenny Barron',
        'Kenny Drew',
        'McCoy Tyner',
        'Oscar Peterson',
        'Ray Bryant',
        'Teddy Wilson',
        'Tete Montoliu',
        'Thelonious Monk',
        'Tommy Flanagan'
    ]
}


def validate_conditions(conditions: list[str] | str) -> None:
    """Validates a single condition or list of conditions, raises ValuError for invalid conditions"""
    # Allows for a single string as input
    if isinstance(conditions, str):
        conditions = [conditions]
    for condition in conditions:
        if condition.lower() not in ACCEPT_CONDITIONS:
            raise ValueError(f'expected `condition` to be in {", ".join(ACCEPT_CONDITIONS)} but got {condition}')


def load_tivo_metadata() -> list[dict]:
    """Load all metadata JSONs from data/root and references/tivo_artist_metadata"""
    metadata_filepaths = []
    # Get the JSON files from data/raw (i.e., for individual tracks)
    for dataset in utils.DATASETS_WITH_TIVO:
        dataset_jsons = [
            j for j in utils.get_data_files_with_ext(
                dir_from_root=os.path.join("data/raw", dataset), ext="**/*.json"
            )
        ]
        metadata_filepaths.extend([j for j in dataset_jsons if j.endswith("_tivo.json")])
    # Get the JSON files from references/tivo_artist_metadata (i.e., for each artist)
    for artist in os.listdir(os.path.join(utils.get_project_root(), "references/tivo_artist_metadata")):
        metadata_filepaths.append(os.path.join(utils.get_project_root(), "references/tivo_artist_metadata", artist))
    # Iterate through and read all the JSONs (with a cache)
    for file in metadata_filepaths:
        yield utils.read_json_cached(file)


def get_inner_json_values(metadata: dict, key: str):
    """Get values from possibly-nested dictionary according to given key"""
    condition_val = metadata[key]
    # Genre, mood, and themes are all lists of dictionaries
    res = []
    if isinstance(condition_val, list):
        for condition_val_val in condition_val:
            res.append(condition_val_val['name'])  # we also have weight values here, but we're not getting them?
    # Performer is just a single string value
    else:
        res.append(condition_val)
    yield from res


def validate_condition_values(
        condition_values: list[tuple[str, int]],
        condition_name: str
) -> list[tuple[str, int]]:
    """Validates values for a given condition by merging similar entries, removing invalid ones, etc."""
    validated = {}
    for value, weight in condition_values:
        # Skip over values that we don't want to use
        if value in EXCLUDE[condition_name]:
            continue
        else:
            # Merge a value with its "master" key (i.e., Show Tunes -> Stage & Screen, Soundtrack -> Stage & Screen)
            for merge_key, merge_values in MERGE[condition_name].items():
                if value in merge_values:
                    value = merge_key
            # This ensures that we only store the HIGHEST weight for any value
            if value not in validated.keys() or weight > validated[value]:
                validated[value] = weight
    # Sanity check that none of our values should now be duplicates
    assert len(set(validated.keys())) == len(validated.keys())
    # Sort the values by their weight, in descending order (highest weight first)
    return sorted(list(validated.items()), key=lambda x: x[1], reverse=True)


def _get_pianist_genres(pianist_name: str) -> list[tuple[str, int]]:
    """Get the genres & weights associated with the PIANIST playing on a track (not the track itself)"""
    pianist_metadata = os.path.join(
        utils.get_project_root(),
        "references/tivo_artist_metadata",
        pianist_name.replace(" ", "") + ".json"
    )
    # If we have metadata for the pianist, grab the associated genres
    if os.path.isfile(pianist_metadata):
        # Read the metadata for the pianist
        pianist_metadata_dict = utils.read_json_cached(pianist_metadata)
        # If we have genres for the pianist
        if len(pianist_metadata_dict["genres"]) > 0:
            return [(x["name"], x["weight"]) for x in pianist_metadata_dict["genres"]]
        # Otherwise, return an empty list
        else:
            return []
    # This will trigger if we SHOULD have metadata for the current pianist, but we can't find the file
    elif pianist_name not in EXCLUDE["pianist"]:
        raise FileNotFoundError(f"Could not find metadata file at {pianist_metadata}!")
    # This will trigger if we shouldn't have metadata for the pianist: silently return an empty list
    else:
        return []


def _get_track_genres(track_metadata_dict: dict) -> list[tuple[str, int]]:
    """Get the genres & weights associated with a track"""
    # Grab the genres associated with the track
    if len(track_metadata_dict) > 0:
        return [(x["name"], x["weight"]) for x in track_metadata_dict["genres"]]
    else:
        return []


def get_genre_tokens(
        track_metadata_dict: dict,
        tokenizer: MusicTokenizer,
        n_genres: int = MAX_GENRE_TOKENS_PER_TRACK
) -> list[str]:
    """Gets tokens for a track's genres: either from the track itself, or (if none found) from the artist"""
    # Check that we've added pianist tokens to our tokenizer
    assert len([i for i in tokenizer.vocab.keys() if "GENRES" in i]) > 0, "Genre tokens not added to tokenizer!"
    # Get the genres from the track and from the pianist
    genres_track = _get_track_genres(track_metadata_dict)
    genres_pianist = _get_pianist_genres(track_metadata_dict["pianist"])
    # Merge them together
    genres = genres_track + genres_pianist
    if len(genres) == 0:
        return []
    # Run validation: this will remove duplicates and sort depending on weight
    validated_genres = validate_condition_values(genres, "genres")
    # Remove the weight term from each tuple to get a single list
    finalised_genres = [g[0] for g in validated_genres]
    # Subset to only get the top-N genres, if required
    if n_genres is not None:
        finalised_genres = finalised_genres[:n_genres]
    # Add the prefix to the token
    prefixed = [f'GENRES_{utils.remove_punctuation(g).replace(" ", "")}' for g in finalised_genres]
    # Sanity checks
    assert len(set(prefixed)) == len(prefixed)
    for pfix in prefixed:
        assert pfix in tokenizer.vocab.keys(), f"Could not find token {pfix} in tokenizer vocabulary!"
    return prefixed


def _get_similar_pianists(pianist_name: str) -> list[tuple[str, int]]:
    """Get names + weights for pianists SIMILAR to the current pianist on a track"""
    pianist_metadata = os.path.join(
        utils.get_project_root(),
        "references/tivo_artist_metadata",
        pianist_name.replace(" ", "") + ".json"
    )
    # If we have metadata for the pianist, grab the other pianists that TiVo says they are similar to
    if os.path.isfile(pianist_metadata):
        pianist_metadata_dict = utils.read_json_cached(pianist_metadata)
        all_pianists = [(x["name"], x["weight"]) for x in pianist_metadata_dict["similars"]]
        return validate_condition_values(all_pianists, "pianist")
    # This will trigger if we SHOULD have metadata for the current pianist, but we can't find the file
    elif pianist_name not in EXCLUDE["pianist"]:
        raise FileNotFoundError(f"Could not find metadata file at {pianist_metadata}!")
    # This will trigger if we DON'T have metadata for the current pianist, and we SHOULDN't have metadata
    else:
        return []


def get_pianist_tokens(
        track_metadata_dict: dict,
        tokenizer: MusicTokenizer,
        n_pianists: int = MAX_SIMILAR_PIANISTS
) -> list[str]:
    # Check that we've added pianist tokens to our tokenizer
    assert len([i for i in tokenizer.vocab.keys() if "PIANIST" in i]) > 0, "Pianist tokens not added to tokenizer!"
    # Get the pianist FROM THIS TRACK
    track_pianist = track_metadata_dict["pianist"]
    # If we want to use this pianist
    if track_pianist not in EXCLUDE["pianist"]:
        finalised_pianists = [track_pianist]
    else:
        similar_pianists = _get_similar_pianists(track_pianist)
        if n_pianists is not None:
            similar_pianists = similar_pianists[:n_pianists]
        # Subset to remove weight
        finalised_pianists = [i[0] for i in similar_pianists]
    # Add the prefix to the token
    prefixed = [f'PIANIST_{utils.remove_punctuation(g).replace(" ", "")}' for g in finalised_pianists]
    # Sanity check that the tokens are part of the vocabulary for the tokenizer
    for pfix in prefixed:
        assert pfix in tokenizer.vocab.keys(), f"Could not find token {pfix} in tokenizer vocabulary!"
    return prefixed


def get_tempo_token(tempo: float, tokenizer: MusicTokenizer, _raise_on_difference_exceeding: int = 50) -> str:
    """Given a tempo for a track, get the closest tempo token from the tokenizer"""
    # Get the tempo tokens from the tokenizer
    tempo_tokens = [i for i in tokenizer.vocab.keys() if "TEMPOCUSTOM" in i]
    assert len(tempo_tokens) > 0, "Custom tempo tokens not added to tokenizer!"
    # Get tempo values as integers, rather than strings
    tempo_stripped = np.array([int(i.replace("TEMPOCUSTOM_", "")) for i in tempo_tokens])
    # Get the difference between the passed tempo and the tempo tokens used by the tokenizer
    sub = np.abs(tempo - tempo_stripped)
    # Raise an error if the closest tempo token is too far away from the actual token
    if np.min(sub) > _raise_on_difference_exceeding:
        raise ValueError(f"Closest tempo token is too far from passed tempo! "
                         f"Got tempo {tempo:.3f}, smallest difference with passed array is {np.min(sub):.3f}")
    # Get the actual tempo token
    tempo_token = tempo_tokens[np.argmin(sub)]
    # Sanity check
    assert "TEMPOCUSTOM" in tempo_token
    # Return the idx of the token, ready to be added to the sequence
    return tempo_token


def get_time_signature_token(time_signature: int, tokenizer: MusicTokenizer) -> str:
    # Get the time signature tokens from the tokenizer
    timesig_tokens = [i for i in tokenizer.vocab.keys() if "TIMESIGNATURECUSTOM" in i]
    assert len(timesig_tokens) > 0, "Custom time signature tokens not added to tokenizer!"
    # Get the corresponding token
    timesig_token = f'TIMESIGNATURECUSTOM_{time_signature}4'
    # Return the idx, ready to be added to the sequence
    if timesig_token in timesig_tokens:
        return timesig_token
    # Raise an error if the token isn't in the vocabulary (should never happen)
    else:
        raise AttributeError(f"Tokenizer does not have token {timesig_token} in vocabulary!")


if __name__ == "__main__":
    from collections import Counter

    from miditok import MIDILike
    from jazz_style_conditioned_generation.data.tokenizer import add_genres_to_vocab, add_pianists_to_vocab

    tokfactory = MIDILike()
    js_fps = utils.get_data_files_with_ext("data/raw", "**/*_tivo.json")
    add_genres_to_vocab(tokfactory)
    add_pianists_to_vocab(tokfactory)

    track_genres, track_pianists = [], []
    for js in js_fps:
        js_loaded = utils.read_json_cached(js)
        track_genres.extend(get_genre_tokens(js_loaded, tokfactory, n_genres=5))
        track_pianists.extend(get_pianist_tokens(js_loaded, tokfactory, n_pianists=5))

    print("Loaded", len(set(track_genres)), "genres")
    assert len(set(track_genres)) == len([i for i in tokfactory.vocab.keys() if "GENRES" in i])
    print("Genre counts: ", Counter(track_genres))
    print("Loaded", len(set(track_pianists)), "pianists")
    assert len(set(track_pianists)) == len([i for i in tokfactory.vocab.keys() if "PIANIST" in i])
    print("Pianist counts: ", Counter(track_pianists))
