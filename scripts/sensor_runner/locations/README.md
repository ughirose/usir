# locations/ — 作品ごとの場所リスト（手で確定したものだけ）

`<work_id>.json`（schema `sensor.locations/1`）がある作品でのみ、直前場面の場所が `location_id` として L2 鍵に入る。
無い作品では location は鍵に入らない（自由文の location は鍵に入れない）。

- リストは人が作る。計測データ（Step 0 / 0.5 の観測値）から自動生成して置かない。
- `none`（場所なし）は常に正規の値で、エントリは不要。
- 本文が `label` / `aliases`（NFKC・空白・記号を除いて比較）のどれにも一致しない非空の場所は `unlisted` になる。
  `none` とは区別され、`aliases` の不足が見える。

```json
{"schema": "sensor.locations/1", "work_id": "example",
 "locations": [{"id": "L1", "label": "下宿", "aliases": ["自分の部屋"]}]}
```

現状、本番のリストは置いていない（場所リストの確定は人の作業）。
