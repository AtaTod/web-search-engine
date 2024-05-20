import json
from nltk.stem.snowball import SnowballStemmer
import os
import re
import sys
import math
from pathlib import Path
from typing import Optional, List, Union, Dict
import pickle


class SAR_Indexer:
    """
    Prototipo de la clase para realizar la indexacion y la recuperacion de artículos de Wikipedia
        
        Preparada para todas las ampliaciones:
          parentesis + multiples indices + posicionales + stemming + permuterm

    Se deben completar los metodos que se indica.
    Se pueden añadir nuevas variables y nuevos metodos
    Los metodos que se añadan se deberan documentar en el codigo y explicar en la memoria
    """

    # lista de campos, el booleano indica si se debe tokenizar el campo
    # NECESARIO PARA LA AMPLIACION MULTIFIELD
    fields = [
        ("all", True), ("title", True), ("summary", True), ("section-name", True), ('url', False),
    ]
    def_field = 'all'
    PAR_MARK = '%'
    # numero maximo de documento a mostrar cuando self.show_all es False
    SHOW_MAX = 10

    all_atribs = ['urls', 'index', 'sindex', 'ptindex', 'docs', 'weight', 'articles',
                  'tokenizer', 'stemmer', 'show_all', 'use_stemming']

    queryOperationsRegex = r'[\w-]+|AND|OR|NOT|\(|\)'

    def __init__(self):
        """
        Constructor de la classe SAR_Indexer.
        NECESARIO PARA LA VERSION MINIMA

        Incluye todas las variables necesaria para todas las ampliaciones.
        Puedes añadir más variables si las necesitas 

        """
        self.nextdocid = 0
        self.nextartid = 0
        self.urls = set() # hash para las urls procesadas,
        self.index = {} # hash para el indice invertido de terminos --> clave: termino, valor: posting list
        self.sindex = {} # hash para el indice invertido de stems --> clave: stem, valor: lista con los terminos que tienen ese stem
        self.ptindex = {} # hash para el indice permuterm.
        self.docs = {} # diccionario de terminos --> clave: entero(docid),  valor: ruta del fichero.
        self.weight = {} # hash de terminos para el pesado, ranking de resultados.
        self.articles = {} # hash de articulos --> clave entero (artid), valor: la info necesaria para diferencia los artículos dentro de su fichero
        self.tokenizer = re.compile("\W+") # expresion regular para hacer la tokenizacion
        self.stemmer = SnowballStemmer('spanish') # stemmer en castellano
        self.show_all = False # valor por defecto, se cambia con self.set_showall()
        self.show_snippet = False # valor por defecto, se cambia con self.set_snippet()
        self.use_stemming = False # valor por defecto, se cambia con self.set_stemming()
        self.use_ranking = False  # valor por defecto, se cambia con self.set_ranking()

    ###############################
    ###                         ###
    ###      CONFIGURACION      ###
    ###                         ###
    ###############################

    def set_showall(self, v: bool):
        """

        Cambia el modo de mostrar los resultados.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_all es True se mostraran todos los resultados el lugar de un maximo de self.SHOW_MAX, no aplicable a la opcion -C

        """
        self.show_all = v

    def set_snippet(self, v: bool):
        """

        Cambia el modo de mostrar snippet.
        
        input: "v" booleano.

        UTIL PARA TODAS LAS VERSIONES

        si self.show_snippet es True se mostrara un snippet de cada noticia, no aplicable a la opcion -C

        """
        self.show_snippet = v

    def set_stemming(self, v: bool):
        """

        Cambia el modo de stemming por defecto.
        
        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v

    #############################################
    ###                                       ###
    ###      CARGA Y GUARDADO DEL INDICE      ###
    ###                                       ###
    #############################################

    def save_info(self, filename: str):
        """
        Guarda la información del índice en un fichero en formato binario
        
        """
        info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'wb') as fh:
            pickle.dump(info, fh)

    def load_info(self, filename: str):
        """
        Carga la información del índice desde un fichero en formato binario
        
        """
        # info = [self.all_atribs] + [getattr(self, atr) for atr in self.all_atribs]
        with open(filename, 'rb') as fh:
            info = pickle.load(fh)
        atrs = info[0]
        for name, val in zip(atrs, info[1:]):
            setattr(self, name, val)

    ###############################
    ###                         ###
    ###   PARTE 1: INDEXACION   ###
    ###                         ###
    ###############################

    def already_in_index(self, article: Dict) -> bool:
        """

        Args:
            article (Dict): diccionario con la información de un artículo

        Returns:
            bool: True si el artículo ya está indexado, False en caso contrario
        """
        return article['url'] in self.urls

    def index_dir(self, root: str, **args):
        """
        
        Recorre recursivamente el directorio o fichero "root" 
        NECESARIO PARA TODAS LAS VERSIONES
        
        Recorre recursivamente el directorio "root"  y indexa su contenido
        los argumentos adicionales "**args" solo son necesarios para las funcionalidades ampliadas

        """
        self.multifield = args['multifield']
        self.positional = args['positional']
        self.stemming = args['stem']
        self.permuterm = args['permuterm']

        file_or_dir = Path(root)

        if file_or_dir.is_file():
            # is a file
            self.generate_docid()
            self.index_file(root)
        elif file_or_dir.is_dir():
            # is a directory
            for d, _, files in os.walk(root):
                for filename in sorted(files):
                    if filename.endswith('.json'):
                        fullname = os.path.join(d, filename)
                        self.generate_docid()
                        self.index_file(fullname)
        else:
            print(f"ERROR:{root} is not a file nor directory!", file=sys.stderr)
            sys.exit(-1)

        ##########################################
        ## COMPLETAR PARA FUNCIONALIDADES EXTRA ##
        ##########################################

    def parse_article(self, raw_line: str) -> Dict[str, str]:
        """
        Crea un diccionario a partir de una linea que representa un artículo del crawler

        Args:
            raw_line: una linea del fichero generado por el crawler

        Returns:
            Dict[str, str]: claves: 'url', 'title', 'summary', 'all', 'section-name'
        """

        article = json.loads(raw_line)
        sec_names = []
        txt_secs = ''
        for sec in article['sections']:
            txt_secs += sec['name'] + '\n' + sec['text'] + '\n'
            txt_secs += '\n'.join(
                subsec['name'] + '\n' + subsec['text'] + '\n' for subsec in sec['subsections']) + '\n\n'
            sec_names.append(sec['name'])
            sec_names.extend(subsec['name'] for subsec in sec['subsections'])
        article.pop('sections')  # no la necesitamos
        article['all'] = article['title'] + '\n\n' + article['summary'] + '\n\n' + txt_secs
        article['section-name'] = '\n'.join(sec_names)

        return article

    def generate_artid(self):
        artid = self.nextartid
        self.nextartid += 1
        return artid

    def generate_docid(self):
        docid = self.nextdocid
        self.nextdocid += 1
        return docid

    def index_file(self, filename: str):
        """

        Indexa el contenido de un fichero.
        
        input: "filename" es el nombre de un fichero generado por el Crawler cada línea es un objeto json
            con la información de un artículo de la Wikipedia

        NECESARIO PARA TODAS LAS VERSIONES

        dependiendo del valor de self.multifield y self.positional se debe ampliar el indexado


        """
        for i, line in enumerate(open(filename)):
            article = self.parse_article(line)
            if self.already_in_index(article):
                continue

            self.urls.add(article['url'])
            artid = self.generate_artid()
            self.articles[artid] = (self.nextdocid - 1, article['url'])

            if self.multifield:
                if self.positional:
                    self.index_pos(article, 'url', artid)
                    self.index_pos(article, 'title', artid)
                    self.index_pos(article, 'summary', artid)
                    self.index_pos(article, 'all', artid)
                    self.index_pos(article, 'section-name', artid)
                else:
                    self.index_no_pos(article, 'url', artid)
                    self.index_no_pos(article, 'title', artid)
                    self.index_no_pos(article, 'summary', artid)
                    self.index_no_pos(article, 'all', artid)
                    self.index_no_pos(article, 'section-name', artid)
            else:
                if self.positional:
                    self.index_pos(article, 'all', artid)
                else:
                    self.index_no_pos(article, 'all', artid)

    def index_pos(self, article: dict, field, artid: int):
        for pos, token in enumerate(self.tokenize(article[field])):
            if field in self.index:
                if token in self.index[field]:
                    if artid in self.index[field][token]:
                        self.index[field][token][artid].append(pos)
                    else:
                        self.index[field][token][artid] = [pos]
                else:
                    self.index[field][token] = {artid: [pos]}
            else:
                self.index[field] = {token: {artid: [pos]}}

    def index_no_pos(self, article: dict, field, artid: int):
        for pos, token in enumerate(self.tokenize(article[field])):
            if field in self.index:
                if token in self.index[field]:
                    if artid in self.index[field][token]:
                        self.index[field][token][artid] += 1
                    else:
                        self.index[field][token][artid] = 1
                else:
                    self.index[field][token] = {artid: 1}
            else:
                self.index[field] = {token: {artid: 1}}




    def set_stemming(self, v: bool):
        """

        Cambia el modo de stemming por defecto.

        input: "v" booleano.

        UTIL PARA LA VERSION CON STEMMING

        si self.use_stemming es True las consultas se resolveran aplicando stemming por defecto.

        """
        self.use_stemming = v

    def tokenize(self, text: str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Tokeniza la cadena "texto" eliminando simbolos no alfanumericos y dividientola por espacios.
        Puedes utilizar la expresion regular 'self.tokenizer'.

        params: 'text': texto a tokenizar

        return: lista de tokens

        """
        return self.tokenizer.sub(' ', text.lower()).split()

    def make_stemming(self):
        """

        Crea el indice de stemming (self.sindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE STEMMING.

        "self.stemmer.stem(token) devuelve el stem del token"


        """

        pass
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

    def make_permuterm(self):
        """

        Crea el indice permuterm (self.ptindex) para los terminos de todos los indices.

        NECESARIO PARA LA AMPLIACION DE PERMUTERM


        """
        pass
        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

    def show_stats(self):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Muestra estadisticas de los indices

        """
        print('========================================')
        print(f'Number of indexed files: {self.nextdocid}')
        print('----------------------------------------')
        print(f'Number of indexed articles: {self.nextartid}')
        print('----------------------------------------')
        print('TOKENS:')
        if self.multifield:
            for field in {'all', 'title', 'summary', 'section-name', 'url'}:
                print(f'\t# of tokens in \'{field}\': {len(self.index[field])}')
        else:
            field = 'all'
            print(f'\t# of tokens in \'{field}\': {len(self.index[field])}')
        print('----------------------------------------')
        if self.positional:
            print('Positional queries are allowed.')
        else:
            print('Positional queries are not allowed.')
        print('========================================')





    #################################
    ###                           ###
    ###   PARTE 2: RECUPERACION   ###
    ###                           ###
    #################################

    ###################################
    ###                                 ###
    ###   PARTE 2.1: RECUPERACION   ###
    ###                             ###
    ###################################

    def solve_query(self, query: str, prev: Dict = {}):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una query.
        Debe realizar el parsing de consulta que sera mas o menos complicado en funcion de la ampliacion que se implementen


        param:  "query": cadena con la query
                "prev": incluido
                por si se quiere hacer una version recursiva. No es necesario utilizarlo.


        return: posting list con el resultado de la query

        """

        tokens = re.findall(self.queryOperationsRegex, query)
        postfix = self.infix_to_postfix(tokens)
        return self.evaluate_postfix(postfix)

    def infix_to_postfix(self, tokens):
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1, '(': 0, ')': 0}
        output = []
        operator_stack = []

        for token in tokens:
            if token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                operator_stack.pop()  # Descartar el paréntesis izquierdo
            elif token in precedence:
                while operator_stack and precedence[operator_stack[-1]] >= precedence[token]:
                    output.append(operator_stack.pop())
                operator_stack.append(token)
            else:
                output.append(token)

        while operator_stack:
            output.append(operator_stack.pop())

        return output

    def evaluate_postfix(self, postfix):
        stack = []



        for token in postfix:
            if token not in ['AND', 'OR', 'NOT']:
                stack.append(self.get_posting(token))
            else:
                if token == 'NOT':
                    op1 = stack.pop()
                    result = self.reverse_posting(op1)
                else:
                    op2 = stack.pop()
                    op1 = stack.pop()
                    if token == 'AND':
                        result = self.and_posting(op1, op2)
                    elif token == 'OR':
                        result = self.or_posting(op1, op2)
                stack.append(result)

        return stack.pop()





    def get_posting(self, term:str, field:Optional[str]=None):
        """

        Devuelve la posting list asociada a un termino.
        Dependiendo de las ampliaciones implementadas "get_posting" puede llamar a:
            - self.get_positionals: para la ampliacion de posicionales
            - self.get_permuterm: para la ampliacion de permuterms
            - self.get_stemming: para la amplaicion de stemming


        param:  "term": termino del que se debe recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario si se hace la ampliacion de multiples indices

        return: posting list

        NECESARIO PARA TODAS LAS VERSIONES

        """
        if field is None:
            field = self.def_field

        if term in self.index.get(field, {}):
            return list(self.index[field][term].keys())
        else:
            return []

        def flatten(nested_list):
            for item in nested_list:
                if isinstance(item, list):
                    yield from flatten(item)
                else:
                    yield item

        if field == None:
            field = ['all']

        postings_lists = []

        for field in field:
            postings_lists.append(list(self.index[field][term].keys()))

        print(postings_lists[0])

        return postings_lists[0]


    def get_positionals(self, terms: str, index):
        """

        Devuelve la posting list asociada a una secuencia de terminos consecutivos.
        NECESARIO PARA LA AMPLIACION DE POSICIONALES

        param:  "terms": lista con los terminos consecutivos para recuperar la posting list.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """
        #TODO
        pass
        ########################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE POSICIONALES ##
        ########################################################

    def get_stemming(self, term: str, field: Optional[str] = None):
        """

        Devuelve la posting list asociada al stem de un termino.
        NECESARIO PARA LA AMPLIACION DE STEMMING

        param:  "term": termino para recuperar la posting list de su stem.
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        stem = self.stemmer.stem(term)

        ####################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA DE STEMMING ##
        ####################################################

    def get_permuterm(self, term: str, field: Optional[str] = None):
        """

        Devuelve la posting list asociada a un termino utilizando el indice permuterm.
        NECESARIO PARA LA AMPLIACION DE PERMUTERM

        param:  "term": termino para recuperar la posting list, "term" incluye un comodin (* o ?).
                "field": campo sobre el que se debe recuperar la posting list, solo necesario se se hace la ampliacion de multiples indices

        return: posting list

        """

        ##################################################
        ## COMPLETAR PARA FUNCIONALIDAD EXTRA PERMUTERM ##
        ##################################################
        pass

    def reverse_posting(self, p: list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Devuelve una posting list con todas las noticias excepto las contenidas en p.
        Util para resolver las queries con NOT.


        param:  "p": posting list


        return: posting list con todos los artid exceptos los contenidos en p

        """

        ########################################
        ## COMPLETAR PARA TODAS LAS VERSIONES ##
        ########################################

        all_post = list(self.articles.keys())
        p_set = set(p)
        return [doc for doc in all_post if doc not in p_set]




    def and_posting(self, p1:list, p2:list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el AND de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos en p1 y p2

        """
        result = []
        i, j = 0, 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                result.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                i += 1
            else:
                j += 1
        return result


        p1_len = len(p1)
        p2_len = len(p2)
        p1_pun, p2_pun = 0
        out = []
        while p1_pun < p1_len and p2_pun < p2_len:
            if p1[p1_pun] == p2[p2_pun]:
                out.append(p1)
                p1_pun += 1
                p2_pun += 1
            else:
                if p1[p1_pun] < p2[p2_pun]:
                    p1_pun += 1
                else:
                    p2_pun += 1
        return out

    def or_posting(self, p1: list, p2: list):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Calcula el OR de dos posting list de forma EFICIENTE

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 o p2

        """

        result = []
        i, j = 0, 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                result.append(p1[i])
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                result.append(p1[i])
                i += 1
            else:
                result.append(p2[j])
                j += 1
        result.extend(p1[i:])
        result.extend(p2[j:])
        return result




    def minus_posting(self, p1, p2):
        """
        OPCIONAL PARA TODAS LAS VERSIONES

        Calcula el except de dos posting list de forma EFICIENTE.
        Esta funcion se incluye por si es util, no es necesario utilizarla.

        param:  "p1", "p2": posting lists sobre las que calcular


        return: posting list con los artid incluidos de p1 y no en p2

        """
        result = []
        i, j = 0, 0
        while i < len(p1) and j < len(p2):
            if p1[i] == p2[j]:
                i += 1
                j += 1
            elif p1[i] < p2[j]:
                result.append(p1[i])
                i += 1
            else:
                j += 1
        result.extend(p1[i:])
        return result





    #####################################
    ###                               ###
    ### PARTE 2.2: MOSTRAR RESULTADOS ###
    ###                               ###
    #####################################

    def solve_and_count(self, ql: List[str], verbose: bool = True) -> List:
        results = []
        for query in ql:
            if len(query) > 0 and query[0] != '#':
                r = self.solve_query(query)
                results.append(len(r))
                if verbose:
                    print(f'{query}\t{len(r)}')
            else:
                results.append(0)
                if verbose:
                    print(query)
        return results

    def solve_and_test(self, ql: List[str]) -> bool:
        errors = False
        for line in ql:
            if len(line) > 0 and line[0] != '#':
                query, ref = line.split('\t')
                reference = int(ref)
                result = len(self.solve_query(query))
                if reference == result:
                    print(f'{query}\t{result}')
                else:
                    print(f'>>>>{query}\t{reference} != {result}<<<<')
                    errors = True
            else:
                print(line)
        return not errors

    def solve_and_show(self, query: str):
        """
        NECESARIO PARA TODAS LAS VERSIONES

        Resuelve una consulta y la muestra junto al numero de resultados

        param:  "query": query que se debe resolver.

        return: el numero de artículo recuperadas, para la opcion -T

        """
        pass
        ################
        ## COMPLETAR  ##
        ################
