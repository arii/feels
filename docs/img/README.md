# Rendered diagrams

Generated artifacts — **don't hand-edit**. Each PNG is rendered from a
```` ```mermaid ```` block that lives in a markdown file; that block is the source
of truth. GitHub renders those blocks natively, so these files exist only for
slides and decks.

| file | source block |
|---|---|
| `architecture.png` | [`../../README.md`](../../README.md) → *Architecture*, 1st block (system overview) |
| `model-architecture.png` | [`../05-ml-pipeline.md`](../05-ml-pipeline.md) → *0. Two models, one recording* |

## Regenerate

Needs Node and a local Chrome (reused so puppeteer doesn't download its own):

```bash
mkdir -p .render
echo '{"executablePath":"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}' > .render/pptr.json
( cd .render && PUPPETEER_SKIP_DOWNLOAD=true npm install @mermaid-js/mermaid-cli@11 )

python3 - <<'PY'
import re
def block(path, i=0):
    return re.findall(r"```mermaid\n(.*?)```", open(path).read(), re.S)[i]
open(".render/architecture.mmd", "w").write(block("README.md"))
open(".render/model-architecture.mmd", "w").write(block("docs/05-ml-pipeline.md"))
PY

for n in architecture model-architecture; do
  .render/node_modules/.bin/mmdc -i .render/$n.mmd -o docs/img/$n.png \
    -p .render/pptr.json -b white -s 2
done

rm -rf .render   # ~465 MB of node_modules
```
