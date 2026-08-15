# InboxGuardian Evaluation Labeling Policy

## Junk vs. protected email

InboxGuardian is optimizing for **personal inbox cleanup**, not generic spam detection.

`junk` means an email that the user considers unwanted and safe to remove from the inbox. It does not necessarily mean the sender is malicious or that Gmail would classify it as spam.

Examples for this user:

- Job alerts are junk because the user is not currently applying for jobs.
- Newsletters such as Satvic Movement and news/editor newsletters are junk.
- Insurance marketing/renewal emails from providers for which the user has no policy are junk.
- Course and LLM training promotions are junk.

## Protected email

Protected email is information that must not be autonomously deleted, including:

- Job offers and employment documents, including from previous employers
- Salary slips
- Investment statements and confirmations
- Insurance/policy documents and premium receipts
- Tax and legal records
- Personal correspondence
- Work correspondence
- Banking, payment, transaction and security records

## Ambiguous cases

If the system cannot establish that an email is safe to remove, it must route the case to **HUMAN REVIEW** rather than delete it.

## Important distinction

The same sender or domain can produce both junk and protected email. Classification must therefore consider **email intent and information value**, not sender identity alone.

Examples:

- LIC premium receipt -> KEEP
- Policybazaar insurance advertisement -> DELETE
- Previous employer offer letter -> KEEP
- Job alert from Naukri -> DELETE for this user
