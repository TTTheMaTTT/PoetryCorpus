# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Класс рифм.

import os
import pickle
import sys
import xml.etree.ElementTree as etree

from poetry.apps.corpus.scripts.phonetics.phonetics_markup import Markup
from poetry.apps.corpus.scripts.util.preprocess import VOWELS
from poetry.apps.corpus.scripts.util.vocabulary import Vocabulary
from poetry.settings import BASE_DIR


class Rhymes(object):
    """
    Поиск, обработка и хранение рифм.
    """
    def __init__(self):
        self.rhymes = list()
        self.vocabulary = Vocabulary()

    def add_markup(self, markup, border=5):
        """
        Добавление рифмующихся слов из разметки.
        :param markup: разметка.
        :param border: граница по качеству рифмы.
        """
        for line in markup.lines:
            for word in line.words:
                is_added = self.vocabulary.add_word(word)
                if not is_added:
                    continue
                self.rhymes.append(set())
                index = len(self.vocabulary.words) - 1
                for i, words in enumerate(self.rhymes):
                    for j in words:
                        if not Rhymes.is_rhyme(word, self.vocabulary.get_word(j), score_border=border):
                            continue
                        self.rhymes[i].append(index)
                        self.rhymes[index].append(i)

    def save(self, filename):
        """
        Сохранение состояния данных.
        :param filename: путь к модели.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """
        Загрузка состояния данных.
        :param filename: путь к модели.
        """
        with open(filename, "rb") as f:
            rhymes = pickle.load(f)
            self.__dict__.update(rhymes.__dict__)

    def get_word_rhymes(self, word):
        """
        Поиск рифмы для данного слова.
        :param word: слово.
        :return: список рифм.
        """
        rhymes = []
        for i in range(len(self.vocabulary.words)):
            if not Rhymes.is_rhyme(word, self.vocabulary.get_word(i), score_border=5):
                continue
            rhymes.append(self.vocabulary.get_word(i))
        return rhymes

    @staticmethod
    def get_rhyme_profile(word):
        """
        Получение профиля рифмовки (набора признаков для сопоставления).
        :param word: уже акцентуированное слово (Word).
        :return profile: профиль рифмовки.
        """
        # TODO: Переход на фонетическое слово, больше признаков.
        syllable_number = 0
        accented_syllable = ''
        next_syllable = ''
        next_char = ''
        syllables = list(reversed(word.syllables))
        for i in range(len(syllables)):
            syllable = syllables[i]
            if syllable.accent != -1:
                if i != 0:
                    next_syllable = syllables[i - 1].text
                accented_syllable = syllables[i].text
                if syllable.accent + 1 < len(word.text):
                    next_char = word.text[syllable.accent + 1]
                syllable_number = i
                break
        return syllable_number, accented_syllable, next_syllable, next_char

    @staticmethod
    def is_rhyme(word1, word2, score_border=4, syllable_number_border=4):
        """
        Проверка рифмованности 2 слов.
        :param word1: первое слово для проверки рифмы, уже акцентуированное (Word).
        :param word2: второе слово для проверки рифмы, уже акцентуированное (Word).
        :param score_border: граница определния рифмы, чем выше, тем строже совпадение.
        :param syllable_number_border: ограничение на номер слога с конца, на который падает ударение.
        :return result: является рифмой или нет.
        """
        features1 = Rhymes.get_rhyme_profile(word1)
        features2 = Rhymes.get_rhyme_profile(word2)
        count_equality = 0
        for i in range(len(features1[1])):
            for j in range(i, len(features2[1])):
                if features1[1][i] == features2[1][j]:
                    if features1[1][i] in VOWELS:
                        count_equality += 3
                    else:
                        count_equality += 1
        if features1[2] == features2[2] and features1[2] != '' and features2[2] != '':
            count_equality += 2
        elif features1[3] == features2[3] and features1[3] != '' and features2[3] != '':
            count_equality += 1
        return features1[0] == features2[0] and count_equality >= score_border and \
               features1[0] <= syllable_number_border

    @staticmethod
    def get_all_rhymes():
        """
        Получние рифм всего корпуса.
        :return: объект Rhymes.
        """
        dump_filename = os.path.join(BASE_DIR, "datasets", "rhymes.pickle")
        rhymes = Rhymes()
        if os.path.isfile(dump_filename):
            rhymes.load(dump_filename)
        else:
            i = 0
            markups_filename = os.path.join(BASE_DIR, "datasets", "corpus", "markup_dump.xml")
            for event, elem in etree.iterparse(markups_filename, events=['end']):
                if event == 'end' and elem.tag == 'markup':
                    markup = Markup()
                    markup.from_xml(etree.tostring(elem, encoding='utf-8', method='xml'))
                    rhymes.add_markup(markup)
                    i += 1
                    if i % 500 == 0:
                        sys.stdout.write(str(i) + "\n")
                        sys.stdout.flush()
            rhymes.save(dump_filename)
        return rhymes
