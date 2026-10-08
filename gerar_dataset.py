"""Gera um dataset balanceado de 100.000 perguntas sobre bandas reais.

As frases são perguntas sintéticas para recuperação de informação, não relatos
biográficos. Assim, o dataset usa nomes de bandas reais sem inventar fatos
sobre cidade, ano, integrantes, discos ou apresentações.

Uso: python gerar_dataset.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


BANDS = [
    "The Beatles", "Queen", "Led Zeppelin", "Black Sabbath", "The Rolling Stones",
    "Pink Floyd", "Nirvana", "Metallica", "AC/DC", "U2", "Radiohead",
    "Foo Fighters", "Sepultura", "Legião Urbana", "Titãs", "Os Paralamas do Sucesso",
    "Os Mutantes", "Barão Vermelho", "The Clash", "Ramones",
]

# Quatro formas de perguntar pelo mesmo tema. As frases pedem informação e não
# afirmam que cada banda tenha necessariamente um atributo/evento específico.
TOPICS: dict[str, list[str]] = {
    "formacao": [
        "a formação e origem", "o início da trajetória", "como o grupo começou",
        "a história de formação",
    ],
    "vocalista": [
        "os vocais e seus intérpretes", "quem canta no grupo", "a função dos vocalistas",
        "as vozes associadas à banda",
    ],
    "guitarra": [
        "guitarristas e guitarras", "o papel das guitarras", "os músicos de guitarra",
        "os instrumentos de corda da banda",
    ],
    "baixo": [
        "baixistas e linhas de baixo", "o papel do baixo", "os músicos de contrabaixo",
        "as partes de baixo nas músicas",
    ],
    "bateria": [
        "bateristas e bateria", "o papel da percussão", "os músicos de bateria",
        "as partes rítmicas do grupo",
    ],
    "album_estreia": [
        "o álbum de estreia", "o primeiro disco da carreira", "a estreia em álbum",
        "o disco inicial do grupo",
    ],
    "gravacao": [
        "gravações em estúdio", "o processo de gravação", "a produção dos discos",
        "as sessões de gravação",
    ],
    "letras": [
        "as letras das canções", "os temas das composições", "a escrita das músicas",
        "as palavras e histórias das canções",
    ],
    "turne": [
        "turnês e excursões", "as apresentações em diferentes cidades", "a atividade de shows",
        "as viagens para apresentações",
    ],
    "festival": [
        "a participação em festivais", "os festivais de música", "as apresentações em festivais",
        "os eventos musicais da banda",
    ],
    "publico": [
        "o público e os fãs", "a relação com os admiradores", "a recepção da plateia",
        "os ouvintes da banda",
    ],
    "critica": [
        "a crítica musical", "as resenhas dos discos", "a avaliação de jornalistas musicais",
        "a recepção da imprensa especializada",
    ],
    "vendas": [
        "as vendas de discos", "o desempenho comercial", "as certificações musicais",
        "os números de vendas da discografia",
    ],
    "vinil": [
        "os discos de vinil", "as edições em LP", "as prensagens em vinil",
        "os lançamentos para colecionadores",
    ],
    "radio": [
        "a presença nas rádios", "as execuções radiofônicas", "as canções tocadas no rádio",
        "a divulgação em emissoras de rádio",
    ],
    "clipe": [
        "os videoclipes", "os vídeos musicais", "a produção de clipes", "as imagens das canções",
    ],
    "premio": [
        "os prêmios musicais", "as indicações e premiações", "o reconhecimento da indústria musical",
        "as homenagens recebidas pelo grupo",
    ],
    "reuniao": [
        "reuniões de integrantes", "os reencontros da banda", "possíveis retornos do grupo",
        "a reunião de antigos membros",
    ],
    "separacao": [
        "a separação do grupo", "o encerramento das atividades", "as pausas e términos da banda",
        "o fim da trajetória coletiva",
    ],
    "influencia": [
        "a influência sobre outros músicos", "o legado musical do grupo", "as bandas influenciadas",
        "a contribuição para a história do rock",
    ],
    "rock_brasileiro": [
        "rock brasileiro", "cena de rock no Brasil", "bandas brasileiras de rock", "música rock nacional",
    ],
    "rock_britanico": [
        "rock britânico", "cena britânica de rock", "bandas britânicas de rock", "rock do Reino Unido",
    ],
    "rock_alternativo": [
        "rock alternativo", "cena de música alternativa", "vertentes alternativas do rock",
        "estilo alternativo",
    ],
    "hard_rock": [
        "hard rock", "vertentes pesadas do rock", "rock pesado", "estilos de hard rock",
    ],
    "punk_rock": [
        "punk rock", "cena punk", "vertentes punk da música", "punk na música rock",
    ],
}

ENTITY_FORMS = [
    "{band}", "a banda {band}", "o grupo {band}", "o conjunto musical {band}",
    "a trajetória da banda {band}",
]

FRAMES = [
    "Que informações existem sobre {entity} no assunto {topic}?",
    "O que se pode pesquisar sobre {topic} quando o tema é {entity}?",
    "Que fontes tratam de {topic} ao abordar {entity}?",
    "Onde localizar referências de {topic} sobre {entity}?",
    "Quais aspectos de {topic} aparecem em materiais sobre {entity}?",
    "Que dados ajudam a estudar {topic} em conteúdos sobre {entity}?",
    "Há registros sobre {topic} que mencionam {entity}?",
    "Que perguntas podem orientar uma pesquisa de {topic} sobre {entity}?",
    "Como localizar referências de {topic} que citem {entity}?",
    "Que resultados de busca podem apresentar informações sobre {entity} e {topic}?",
]

QUERIES = [
    {"id": "L01", "tipo": "literal", "topico": "formacao", "consulta": "formação e origem da banda"},
    {"id": "L02", "tipo": "literal", "topico": "vocalista", "consulta": "vocalista e vocais do grupo"},
    {"id": "L03", "tipo": "literal", "topico": "album_estreia", "consulta": "álbum de estreia da banda"},
    {"id": "L04", "tipo": "literal", "topico": "turne", "consulta": "turnê e apresentações da banda"},
    {"id": "L05", "tipo": "literal", "topico": "festival", "consulta": "festivais de música da banda"},
    {"id": "L06", "tipo": "literal", "topico": "critica", "consulta": "crítica musical e resenhas dos discos"},
    {"id": "L07", "tipo": "literal", "topico": "vendas", "consulta": "vendas e certificações de discos"},
    {"id": "L08", "tipo": "literal", "topico": "rock_brasileiro", "consulta": "bandas do rock brasileiro"},
    {"id": "S01", "tipo": "semantica", "topico": "formacao", "consulta": "como os integrantes se conheceram e iniciaram o grupo"},
    {"id": "S02", "tipo": "semantica", "topico": "vocalista", "consulta": "quem interpreta as canções e lidera a voz"},
    {"id": "S03", "tipo": "semantica", "topico": "album_estreia", "consulta": "qual foi o primeiro trabalho longo lançado pelo conjunto"},
    {"id": "S04", "tipo": "semantica", "topico": "turne", "consulta": "viagens para tocar em cidades diferentes"},
    {"id": "S05", "tipo": "semantica", "topico": "festival", "consulta": "eventos musicais com vários artistas no mesmo local"},
    {"id": "S06", "tipo": "semantica", "topico": "critica", "consulta": "o que jornalistas especializados acharam dos discos"},
    {"id": "S07", "tipo": "semantica", "topico": "vendas", "consulta": "popularidade dos discos medida pelo mercado"},
    {"id": "S08", "tipo": "semantica", "topico": "rock_brasileiro", "consulta": "grupos que fazem parte da história musical do Brasil"},
]


def generate(output_dir: Path) -> None:
    if len(BANDS) != 20:
        raise ValueError("O gerador espera 20 bandas para manter 100.000 frases.")
    if any(len(phrases) != 4 for phrases in TOPICS.values()):
        raise ValueError("Cada tópico precisa ter exatamente quatro formulações.")

    output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = output_dir / "dataset_bandas_rock_100k.txt"
    csv_path = output_dir / "dataset_bandas_rock_100k.csv"
    query_path = output_dir / "consultas_avaliacao.json"

    seen: set[str] = set()
    sentence_id = 0
    with txt_path.open("w", encoding="utf-8", newline="\n") as txt_file, csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "topico", "frase"])
        for topic, variants in TOPICS.items():
            for band in BANDS:
                for topic_phrase in variants:
                    for entity_template in ENTITY_FORMS:
                        entity = entity_template.format(band=band)
                        for frame in FRAMES:
                            topic_phrase = topic_phrase.removeprefix("a ").removeprefix("o ").removeprefix("as ").removeprefix("os ")
                            sentence = frame.format(entity=entity, band=band, topic=topic_phrase)
                            if sentence in seen:
                                raise RuntimeError(f"Frase repetida: {sentence}")
                            seen.add(sentence)
                            txt_file.write(sentence + "\n")
                            writer.writerow([sentence_id, topic, sentence])
                            sentence_id += 1

    expected = len(TOPICS) * len(BANDS) * 4 * len(ENTITY_FORMS) * len(FRAMES)
    if sentence_id != 100_000 or expected != 100_000:
        raise RuntimeError(f"Esperadas 100.000 frases únicas, geradas {sentence_id}.")
    query_path.write_text(json.dumps(QUERIES, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "fontes_dataset.json").write_text(json.dumps({
        "tipo": "frases sintéticas em forma de pergunta",
        "bandas_reais": BANDS,
        "observacao": "As frases não afirmam fatos biográficos sobre as bandas; os nomes são entidades reais e os tópicos indicam a intenção da pergunta.",
        "combinacao": "25 tópicos × 20 bandas × 4 variantes temáticas × 5 formas de entidade × 10 estruturas de pergunta",
        "licenca": "Texto gerado pelo script deste trabalho; não é reprodução de biografias.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Geradas {sentence_id:,} perguntas únicas sobre {len(BANDS)} bandas reais.")
    print(f"Dataset TXT: {txt_path}")
    print(f"Metadados CSV: {csv_path}")
    print(f"Consultas de avaliação: {query_path}")


if __name__ == "__main__":
    generate(Path("dados"))
