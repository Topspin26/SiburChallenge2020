from bs4 import BeautifulSoup
import re
from translit import translit
import unidecode

import pycountry
import geonamescache

non_alphanum_regex = re.compile("[^0-9a-zA-Zа-яА-ЯёЁ ]+")

def simple_transform(s, del_brackets=True):
    s0 = s
    s = s.lower()
    s = s.replace('ооо', '').replace('ооо', '').replace('oao', '').replace('оао', '')
    s = s.replace('ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ'.lower(), '')
    s = s.replace('ь', '')
    s = list(translit(s))[0]
    s = unidecode.unidecode(s)

    s = s.replace(',', ' ').replace('.', '').replace('*', '') \
        .replace('"', ' ').replace("'", ' ').replace('-', ' ').replace('&', ' ') \
        .replace('\\', '').replace('?', ' ')
    s = s.replace('(', ' (')
    s = s.replace(')', ') ')
    s = s.replace('( ', '(')
    s = s.replace(' )', ')')
    if len(s.strip()) > 0 and s.strip()[0] == '(':
        s = s.strip()[1:]
    if del_brackets:
        s = re.sub("\(.*\)", "", s)
    s = s.replace('(', '').replace(')', '')
    s = re.sub(non_alphanum_regex, '', s)
    #s = legal_re.sub('', s)
    s = ' '.join(s.split())
    return s


def load_legal_countries0(data_dir):
    with open(data_dir.joinpath('legal_entity_types_by_country.txt'), 'r', encoding='utf-8') as fin:
        s = fin.read()
    soup = BeautifulSoup(s, "lxml")

    countries0 = []

    legal = set()
    for e in soup.find_all('h2')[1:]:
        country = e.find('span').text
        countries0.append(country.lower())
        if country == 'See also':
            break
            #     print()
        for ee in e.findNext('ul').find_all('li'):
            v = ee.text.split('(')[0].split(':')[0].split('≈')[0].split('=')[0].split('–')[0]
            for vv in v.split('/'):
                vv = vv.strip()
                legal.add(vv.lower())
                legal.add(vv.lower().replace('.', ''))
    # print()

    legal.add('pvt')
    legal |= {'de', 'international', 'industries', 'industria', 'imp', 'exp'}

    legal_tokens = set()
    for e in legal:
        legal_tokens.add(re.sub(non_alphanum_regex, '', e))
        for t in e.split():
            ut = unidecode.unidecode(t)
            for tt in [t, re.sub(non_alphanum_regex, '', t),
                       ut, re.sub(non_alphanum_regex, '', ut)]:
                if len(tt) > 2 and tt not in legal and tt not in legal_tokens:
                    # print(tt)
                    legal_tokens.add(tt)

    return legal, countries0, legal_tokens


def get_geo(countries0):
    gc = geonamescache.GeonamesCache()

    countries = countries0 + [country.name.lower() for country in pycountry.countries]
    countries.append('usa')
    countries.append('africa')
    countries.append('asia')
    countries.append('europe')
    countries.append('america')
    countries.append('north')
    countries.append('south')
    countries.append('west')
    countries.append('east')
    countries.append('city')
    countries.append('area')

    countries = set(countries)
    print(len(countries))

    for k, v in gc.get_countries().items():
        c = simple_transform(v['name'])
        if c not in countries:
            countries.add(c)
    print(len(countries))

    for k, v in gc.get_us_states().items():
        c = simple_transform(v['name'])
        if c not in countries:
            countries.add(c)
    print(len(countries))

    cities = set()
    for k, v in gc.get_cities().items():
        c = simple_transform(v['name'])
        cities.add(c)
    print(len(cities))

    cities_alt = set()
    for k, v in gc.get_cities().items():
        c = simple_transform(v['name'])
        cities_alt.add(c)
        for e in v['alternatenames']:
            c = simple_transform(e)
            cities_alt.add(e)
    print(len(cities_alt))

    return countries, cities, cities_alt


def multi_str_replace(strings, debug=True):
    re_str = r'\b(?:' + '|'.join(
        [re.escape(s) for s in strings]
    ) + r')(?!\S)'
    if debug:
        print(re_str)
    return re.compile(re_str, re.UNICODE)


def calc_lcs(s1, s2):
    c = []
    for i in range(len(s1) + 1):
        c.append([0] * (len(s2) + 1))
    for i in range(0, len(s1)):
        for j in range(0, len(s2)):
            if s1[i] == s2[j]:
                c[i + 1][j + 1] = c[i][j] + 1
            else:
                c[i + 1][j + 1] = max(c[i][j + 1], c[i + 1][j])
    return c[len(s1)][len(s2)]


def expand_tokens(t):
    res = set(t)
    for i in range(len(t) - 1):
        res.add(t[i] + t[i + 1])
    for i in range(len(t) - 2):
        res.add(t[i] + t[i + 1] + t[i + 2])
    for i in range(len(t) - 3):
        res.add(t[i][0] + t[i + 1][0] + t[i + 2][0])
    for i in range(len(t) - 4):
        res.add(t[i][0] + t[i + 1][0] + t[i + 2][0] + t[i + 3][0])
    return res
