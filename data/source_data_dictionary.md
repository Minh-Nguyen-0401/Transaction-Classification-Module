# x_ctx — Current Transaction (landing)

### `transactions_landing`

| Column                 | Type          | Description                  | Example                |          |
| ---------------------- | ------------- | ---------------------------- | ---------------------- | -------- |
| txn_id                 | STRING        | Unique transaction id        | `TXN_9f2a7d`           |          |
| sender_id              | STRING        | Customer id (FK)             | `CUST_001238`          |          |
| txn_time_utc           | TIMESTAMP     | Event time UTC               | `2025-11-09T03:21:14Z` |          |
| amount                 | DECIMAL(18,2) | Amount                       | `1250.00`              |          |
| currency               | STRING(3)     | ISO 4217                     | `VND`                  |          |
| tranx_type             | ENUM          | App flow type                | `bill_payment`         |          |
| channel                | ENUM          | Transaction channel          | `WEB`                  | `MOBILE` |
| category               | STRING(3)     | Transaction category (label) | `BIL`                  |          |
| msg_content            | STRING        | Raw memo                     | `Tien nha thang 11`    |          |
| recipient_entity_id    | STRING NULL   | Canonical recipient (if any) | `ENT_rcv_7c91`         |          |
| recipient_alias_raw    | STRING        | Raw payee input              | `Nguyen Van A`         |          |
| to_account_number_hash | STRING        | Hashed account               | `h:acc_2b91...`        |          |
| to_bank_code           | STRING        | Bank code                    | `VCB`                  |          |
| merchant_id            | STRING NULL   | Merchant id if merchant      | `MER_10293`            |          |
| geo_cell_h3_8          | STRING NULL   | Event location cell          | `882a100d4bfffff`      |          |

**Notes:**
- `tranx_type` values: `bill_payment`, `cashback`, `loan_repayment`, `mobile_topup`, `opensaving`, `qrcode_payment`, `stock`, `transfer_in`
- `channel` values: `MOBILE`, `WEB`
- `category` values: See `label_map.json` for category codes and descriptions

---

# x_rcv — Recipient Registry & Profiles

### `recipient_entity`

| Column               | Type   | Description    | Example        |            |
| -------------------- | ------ | -------------- | -------------- | ---------- |
| recipient_entity_id  | STRING | Canonical id   | `ENT_rcv_7c91` |            |
| entity_type          | ENUM   | `MERCHANT      | PERSON`        | `MERCHANT` |
| primary_display_name | STRING | Canonical name | `EVN Ha Noi`   |            |

### `recipient_alias`

| Column              | Type   | Description           | Example        |       |         |                |
| ------------------- | ------ | --------------------- | -------------- | ----- | ------- | -------------- |
| recipient_entity_id | STRING | FK → recipient_entity | `ENT_rcv_7c91` |       |         |                |
| alias_text          | STRING | Observed alias        | `Dien luc HN`  |       |         |                |
| alias_type          | ENUM   | `account_name         | account_number | phone | handle` | `account_name` |

### `merchant_profile`

| Column              | Type      | Description            | Example        |
| ------------------- | --------- | ---------------------- | -------------- |
| merchant_id         | STRING    | Merchant id            | `MER_10293`    |
| recipient_entity_id | STRING    | FK → recipient_entity  | `ENT_rcv_7c91` |
| mcc                 | STRING(4) | Merchant Category Code | `4900`         |
| brand_name          | STRING    | Public brand           | `EVN`          |

### `person_profile` *(coarse, privacy-preserving)*

| Column              | Type   | Description           | Example        |               |         |      |            |         |
| ------------------- | ------ | --------------------- | -------------- | ------------- | ------- | ---- | ---------- | ------- |
| person_id           | STRING | Person id             | `PERS_77b1`    |               |         |      |            |         |
| recipient_entity_id | STRING | FK → recipient_entity | `ENT_rcv_aa12` |               |         |      |            |         |
| age_band            | ENUM   | `18-24                | 25-34          | 35-44         | 45-54   | 55+  | unk`       | `25-34` |
| province_code       | STRING | Coarse location       | `HCM`          |               |         |      |            |         |
| occupation_band     | ENUM   | `student              | salaried       | self_employed | retired | unk` | `salaried` |         |

---

# x_snd — Sender Profile (existing internal)

### `customer_profile`

| Column             | Type   | Description     | Example       |               |          |         |        |            |
| ------------------ | ------ | --------------- | ------------- | ------------- | -------- | ------- | ------ | ---------- |
| sender_id          | STRING | Customer id     | `CUST_001238` |               |          |         |        |            |
| kyc_tier           | ENUM   | `L0             | L1            | L2            | L3`      | `L2`    |        |            |
| segment            | ENUM   | `student        | salaried      | self_employed | merchant | retired | other` | `salaried` |
| age_band           | ENUM   | `18-24          | 25-34         | 35-44         | 45-54    | 55+     | unk`   | `25-34`    |
| home_province_code | STRING | Coarse location | `HCM`         |               |          |         |        |            |
| income_band        | ENUM   | `low            | mid           | upper_mid     | high     | unk`    | `mid`  |            |

**Keys & joins (MVP):**

* `transactions_landing.sender_id` → `customer_profile.sender_id`
* `transactions_landing.recipient_entity_id` → `recipient_entity.recipient_entity_id`
* If merchant: `transactions_landing.merchant_id` → `merchant_profile.merchant_id`
* If person: `recipient_entity.recipient_entity_id` → `person_profile.recipient_entity_id`
