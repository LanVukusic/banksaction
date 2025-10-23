# lol ni generated samo chatgpt mi je napisau tabelo LMAO

| Method | Endpoint                   | Description                                     |
| ------ | -------------------------- | ----------------------------------------------- |
| `GET`  | `/transactions`            | List recent transactions                        |
| `GET`  | `/transactions/{id}`       | View details                                    |
| `POST` | `/transactions/{id}/label` | Label a transaction as fraudulent               |
| `GET`  | `/fraud-similarity`        | Compare transactions vs. known fraud embeddings |
