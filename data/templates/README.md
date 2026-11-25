# Templates

One file per relation, `{en: [...], zh: [...], ru: [...], audit: {...}}`. Template `i` is the same question in every
language, and `[X]` is the person's name.

| File | Relation | Templates | Questions | Incomplete sentences |
|---|---|---|---|---|
| `P19_born.json` | born in (Wikidata P19) | 86 | 20 | 66 |
| `P20_died.json` | died in (P20) | 54 | 20 | 34 |
| `P937_worked.json` | worked in (P937) | 87 | 20 | 67 |

**Source.** The English templates come from MyriadLAMA. Claude Haiku 4.5 translated them into Chinese and Russian.
`audit` keeps an automatic check of each template (for example, phrasing that may draw a description instead of a city),
for the English + Chinese set (`zh_pair`) and the English + Russian set (`ru_pair`). Every template is used, whatever
the audit says.

**Two language pairs.** Each fact is trained and tested in English and one second language: Chinese for the `zh` pair,
Russian for the `ru` pair. The English templates are the same in both pairs.

**Questions and incomplete sentences.** 20 templates of each relation are questions ("Where was [X] born?"): templates
0-19 for born in and died in, 28-47 for worked in. The rest are incomplete sentences that the city completes ("[X] spent
their last days in").

**Answer-format suffix.** Every filled prompt ends with a request to answer with the city:
`(Answer with the name of the city.)`, `（请用城市名称回答。）`, `(Ответьте названием города.)`. The filled prompts are
stored in `../injected/prompts/` and `../known_facts/`, exactly as the models saw them.

**Rebuilding.** The translations and the known facts' city names (from Wikidata) came from LLM and API calls, so a
rebuild would not match word for word. The released files are the ones used in the paper.
