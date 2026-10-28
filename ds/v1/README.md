# simplefunction design system v1

Shared tokens and fonts for simplefunction web projects (this site and Moment Select).

- `tokens.css`: colors, type, spacing, radius, shadows and motion. One system, one accent per product: a page or a section sets `data-product` and only `--accent` / `--on-accent` change.
- `fonts/bricolage-grotesque.woff2`: Bricolage Grotesque (Latin subset, width fixed at 100), SIL Open Font License 1.1, no Reserved Font Name.
- `fonts/simplefunction-text.woff2`: a Korean (KS X 1001, 2,350 syllables) + Latin subset of Pretendard 1.3.9 by Kil Hyung-jin, renamed because "Pretendard" is a Reserved Font Name under the SIL Open Font License 1.1 (OFL FAQ 2.6). Same license.
- License texts: `fonts/OFL-bricolage-grotesque.txt`, `fonts/OFL-pretendard.txt`.

Copy this folder into a project as `ds/v1/` instead of loading it from another site, so a change here never reaches a product without that product's own checks. Change a value, bump the version.
