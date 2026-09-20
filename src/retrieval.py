"""Текстовые индексы, статистика взаимодействий и признаки кандидатов."""

import html
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import snowballstemmer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

_STEMMER = snowballstemmer.stemmer("russian")
_TAGS = re.compile(r"<[^>]+>")
_WORDS = re.compile(r"[а-яa-z0-9]+")


def normalize(value):
    if value is None:
        return ""
    text = html.unescape(str(value)).lower().replace("ё", "е")
    return " ".join(_WORDS.findall(_TAGS.sub(" ", text)))


@lru_cache(maxsize=250_000)
def stem(word):
    return _STEMMER.stemWord(word)


def tokens(text):
    return [stem(word) for word in normalize(text).split()]


def stem_text(text):
    return " ".join(tokens(text))


class BM25:
    def __init__(self, max_features=250_000, k1=1.2, b=0.75):
        self.vectorizer = CountVectorizer(
            token_pattern=r"(?u)\b\w+\b", max_features=max_features, dtype=np.float32
        )
        self.k1, self.b = k1, b

    def fit(self, texts):
        counts = self.vectorizer.fit_transform(texts).tocsr()
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        frequencies = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log1p((counts.shape[0] - frequencies + 0.5) / (frequencies + 0.5))
        scale = self.k1 * (1 - self.b + self.b * lengths / max(lengths.mean(), 1))
        # Нормировка BM25 применяется к ненулевым частотам каждого документа.
        counts.data *= (self.k1 + 1) / (
            counts.data + np.repeat(scale, np.diff(counts.indptr))
        )
        counts.data *= idf[counts.indices].astype(np.float32)
        self.matrix = counts.T.tocsr()
        return self

    def scores(self, text):
        query = self.vectorizer.transform([text])
        query.data[:] = 1
        return (query @ self.matrix).toarray().ravel()


class CharacterIndex:
    def fit(self, texts):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=300_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.matrix = self.vectorizer.fit_transform(texts).T.tocsr()
        return self

    def scores(self, text):
        return (self.vectorizer.transform([text]) @ self.matrix).toarray().ravel()


def topk(scores, k):
    """При равных оценках предпочтение отдаётся меньшему индексу корпуса."""
    k = min(k, len(scores))
    if k == 0:
        return np.array([], dtype=np.int32)
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[: k - len(above)]
    chosen = np.concatenate([above, tied])
    return chosen[np.lexsort((chosen, -scores[chosen]))].astype(np.int32)


class History:
    def __init__(self, train, items):
        texts = train.normalized_query.to_numpy()
        self.texts, row = np.unique(texts, return_inverse=True)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=1,
            max_features=250_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts).T.tocsr()
        microcats = sorted(set(items.item_microcat_id) | set(train.item_microcat_id))
        self.microcat_lookup = {x: i for i, x in enumerate(microcats)}
        self.item_microcat = items.item_microcat_id.map(self.microcat_lookup).to_numpy()
        col = train.item_microcat_id.map(self.microcat_lookup).to_numpy()
        self.topics = sparse.csr_matrix(
            (np.ones(len(train), dtype=np.float32), (row, col)),
            shape=(len(self.texts), len(microcats)),
        )
        total = np.asarray(self.topics.sum(axis=1)).ravel()
        self.topics = sparse.diags(1 / np.maximum(total, 1)) @ self.topics
        item_lookup = pd.Series(np.arange(len(items)), index=items.item_id)
        col = train.item_id.map(item_lookup)
        mask = col.notna().to_numpy()
        self.interactions = sparse.csr_matrix(
            (
                np.ones(mask.sum(), dtype=np.float32),
                (row[mask], col[mask].to_numpy(dtype=int)),
            ),
            shape=(len(self.texts), len(items)),
        )
        self.interactions = sparse.diags(1 / np.maximum(total, 1)) @ self.interactions
        self.popularity = (
            items.item_id.map(train.item_id.value_counts())
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )

    def scores(self, text):
        normalized = normalize(text)
        similarities = (
            (self.vectorizer.transform([normalized]) @ self.matrix).toarray().ravel()
        )
        nearest = topk(similarities, 30)
        weights = similarities[nearest] ** 5
        weights[similarities[nearest] < max(0.15, similarities[nearest[0]] * 0.55)] = 0
        weights /= max(weights.sum(), 1e-10)
        distribution = np.asarray(weights @ self.topics[nearest]).ravel()
        item_scores = np.asarray(weights @ self.interactions[nearest]).ravel()
        return (
            distribution[self.item_microcat],
            item_scores,
            float(similarities[nearest[0]]),
        )


class LocationHistory:
    def __init__(self, train, items):
        locations, self.item_to_location = np.unique(
            items.item_location_id, return_inverse=True
        )
        self.size = len(locations)
        lookup = {value: index for index, value in enumerate(locations)}
        counts = train.groupby(["search_location_id", "item_location_id"]).size()
        self.transitions = {}
        for search_location, group in counts.groupby(level=0):
            total = float(group.sum())
            probabilities = group.to_numpy(dtype=np.float32) / total
            entropy = float(-(probabilities * np.log(probabilities)).sum())
            pairs = [
                (lookup[item_location], count / total)
                for (_, item_location), count in group.items()
                if item_location in lookup
            ]
            self.transitions[search_location] = (pairs, entropy)

    def scores(self, location):
        values = np.zeros(self.size, dtype=np.float32)
        pairs, entropy = self.transitions.get(location, ([], 0.0))
        for index, probability in pairs:
            values[index] = probability
        values = values[self.item_to_location]
        maximum = values.max()
        relative = values / max(maximum, 1e-8)
        return values, relative, entropy


FEATURES = [
    "title_bm25",
    "description_bm25",
    "params_bm25",
    "char_cosine",
    "filter_bm25",
    "title_relative",
    "description_relative",
    "params_relative",
    "lexical",
    "lexical_geo",
    "dense_cosine",
    "dense_relative",
    "dense_geo",
    "topic_probability",
    "history_score",
    "history_similarity",
    "same_location",
    "log_distance",
    "geo",
    "log_popularity",
    "rating",
    "log_reviews",
    "log_price",
    "phone_hidden",
    "message_forbidden",
    "query_words",
    "title_words",
    "title_coverage",
    "description_coverage",
    "phrase_in_title",
    "delivery",
    "has_filter",
    "location_known",
    "topic_geo",
    "lexical_topic",
    "dense_topic",
    "location_probability",
    "location_relative",
    "location_entropy",
]


def relative(values):
    return values / max(float(values.max()), 1e-8)


class Retriever:
    def __init__(self, items, train, indexes, embeddings=None, query_embeddings=None):
        self.items = items
        self.ids = items.item_id.to_numpy()
        self.n = len(items)
        self.indexes = indexes
        self.history = History(train, self.items)
        self.location_history = LocationHistory(train, self.items)
        self.locations = self.items.item_location_id.to_numpy()
        self.lat = self.items.item_latitude.to_numpy(dtype=np.float32)
        self.lon = self.items.item_longitude.to_numpy(dtype=np.float32)
        self.centers = (
            self.items.groupby("item_location_id")[["item_latitude", "item_longitude"]]
            .median()
            .to_dict("index")
        )
        self.titles = self.items.item_title_raw.map(normalize).to_numpy()
        self.title_tokens = [set(stem_text(t).split()) for t in self.titles]
        self.descriptions = (
            self.items.item_description_raw.str.slice(0, 4000).map(normalize).to_numpy()
        )
        self.dense = embeddings is not None
        self.embeddings = embeddings
        self.query_embeddings = query_embeddings
        self.static = np.column_stack(
            [
                self.history.popularity,
                self.items.item_rating.fillna(-1),
                np.log1p(self.items.item_rating_reviews_count.fillna(0).clip(lower=0)),
                np.log1p(self.items.item_price.clip(lower=0)),
                self.items.item_is_phone_hidden.astype(float),
                self.items.item_is_message_forbidden.astype(float),
                [len(x) for x in self.title_tokens],
            ]
        ).astype(np.float32)
        self.static[:, 0] = np.log1p(self.static[:, 0])

    def location_scores(self, location):
        same = (self.locations == location).astype(np.float32)
        center = self.centers.get(location)
        if center is None:
            return (
                same,
                np.full(self.n, 1000, dtype=np.float32),
                np.full(self.n, 0.3, dtype=np.float32),
                0,
            )
        lat, lon = center["item_latitude"], center["item_longitude"]
        # Приближённое расстояние в километрах; масштаб долготы зависит от широты.
        distance = np.sqrt(
            ((self.lat - lat) * 111) ** 2
            + ((self.lon - lon) * 111 * np.cos(np.radians(lat))) ** 2
        )
        distance[same == 1] = 0
        geo = np.maximum(same, np.exp(-distance / 35))
        return same, distance, geo, 1

    def retrieve(self, query, allowed=None):
        text = normalize(query.search_query)
        stemmed = stem_text(text)
        title = self.indexes["title"].scores(stemmed)
        description = self.indexes["description"].scores(stemmed)
        params = self.indexes["params"].scores(stemmed)
        char = self.indexes["char"].scores(text)
        filters = self.indexes["params"].scores(
            stem_text(query.search_infm_params_text)
        )
        topic, history, history_similarity = self.history.scores(text)
        same, distance, geo, known = self.location_scores(query.search_location_id)
        location_probability, location_relative, location_entropy = (
            self.location_history.scores(query.search_location_id)
        )
        region = np.maximum(geo, location_relative)
        lexical = 0.45 * relative(title) + 0.35 * relative(description) + 0.20 * char
        lexical_geo = lexical * (0.25 + 0.75 * geo)
        topic_geo = topic * (0.20 + 0.80 * geo)
        lexical_topic = lexical_geo * (0.25 + 0.75 * np.sqrt(topic))
        if self.dense:
            dense = self.embeddings @ self.query_embeddings[query.query_id]
            # Преобразуем косинусную близость перед умножением на географический вес.
            dense_scaled = np.exp((dense - 1) * 12)
            dense_geo = dense_scaled * (0.25 + 0.75 * geo)
            dense_topic = dense_geo * (0.4 + 0.6 * np.sqrt(topic))
        else:
            dense = np.zeros(self.n, dtype=np.float32)
            dense_geo = dense.copy()
            dense_topic = dense.copy()
        channels = {
            "bm25": (title + description * 0.4, 150),
            "lexical": (lexical, 100),
            "lexical_geo": (lexical_geo, 250),
            "char_geo": (char * (0.25 + 0.75 * geo), 100),
            "lexical_topic": (lexical_topic, 150),
            "history": (history * (0.2 + 0.8 * geo), 80),
            "lexical_region": (lexical * (0.2 + 0.8 * region), 150),
        }
        if self.dense:
            channels.update(
                {
                    "dense": (dense, 150),
                    "dense_geo": (dense_geo, 250),
                    "dense_topic": (dense_topic, 150),
                    "dense_region": (dense_scaled * (0.2 + 0.8 * region), 150),
                }
            )

        def choose(values, k):
            if allowed is None:
                return topk(values, k)
            values = values.copy()
            values[~allowed] = -np.inf
            return topk(values, k)

        candidates = np.unique(
            np.concatenate([choose(score, k) for score, k in channels.values()])
        )
        c = candidates
        query_tokens = set(stemmed.split())
        title_coverage = [
            len(query_tokens & self.title_tokens[i]) / max(len(query_tokens), 1)
            for i in c
        ]
        # Покрытие запроса: доля основ слов, найденных в описании как подстроки.
        description_coverage = [
            sum(t in self.descriptions[i] for t in query_tokens)
            / max(len(query_tokens), 1)
            for i in c
        ]
        s = self.static[c]
        features = np.column_stack(
            [
                title[c],
                description[c],
                params[c],
                char[c],
                filters[c],
                relative(title)[c],
                relative(description)[c],
                relative(params)[c],
                lexical[c],
                lexical_geo[c],
                dense[c],
                relative(dense)[c],
                dense_geo[c],
                topic[c],
                history[c],
                np.full(len(c), history_similarity),
                same[c],
                np.log1p(distance[c]),
                geo[c],
                s[:, 0],
                s[:, 1],
                s[:, 2],
                s[:, 3],
                s[:, 4],
                s[:, 5],
                np.full(len(c), len(query_tokens)),
                s[:, 6],
                title_coverage,
                description_coverage,
                [float(text in self.titles[i]) for i in c],
                np.full(len(c), query.search_is_delivery_search),
                np.full(len(c), bool(query.search_infm_params_text)),
                np.full(len(c), known),
                topic_geo[c],
                lexical_topic[c],
                dense_topic[c],
                location_probability[c],
                location_relative[c],
                np.full(len(c), location_entropy),
            ]
        ).astype(np.float32)
        assert features.shape[1] == len(FEATURES)
        baselines = {name: choose(score, 50) for name, (score, _) in channels.items()}
        return candidates, features, baselines
