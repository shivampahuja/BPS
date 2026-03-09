"""
Script to generate rich sample interaction data for EN and CZ.
Covers all 7 PMI domains with realistic verbatims and metadata.
"""
import csv
import json
import random
from datetime import datetime, timedelta, timezone

random.seed(42)

EN_INTERACTIONS = [
    # A. Product Experience - Device Performance
    ("chat", "UK", "en", "My IQOS device keeps overheating after just 5 minutes of use. This is really frustrating! I've only had it for 2 months.", "neg", -0.7, True, ["THEME_DEVICE_HEATING"], ["empathy"]),
    ("email", "US", "en", "The battery on my device drains incredibly fast - barely lasts through 3 uses before needing a recharge. Very disappointing quality.", "neg", -0.6, True, ["THEME_BATTERY_CHARGE"], []),
    ("call", "UK", "en", "My device stopped working completely after 6 months. It just won't turn on anymore. I need a replacement urgently!", "neg", -0.8, True, ["THEME_DEVICE_BREAKAGE"], ["escalation"]),
    ("review", "DE", "en", "Great product, no heating issues at all after 3 months. Works consistently and battery life has been excellent.", "pos", 0.8, False, ["THEME_DEVICE_HEATING"], []),
    ("chat", "IT", "en", "My IQOS device broke within the first week. Screen cracked without any impact. Very poor build quality!", "neg", -0.9, True, ["THEME_DEVICE_BREAKAGE"], ["escalation"]),
    ("call", "US", "en", "Device heating up much more than before after the last firmware update. Has anyone else experienced this?", "neg", -0.4, False, ["THEME_DEVICE_HEATING"], []),

    # A. Product Experience - Consumables
    ("chat", "UK", "en", "The HEETS tobacco sticks are completely out of stock in all nearby stores and online. When will they be back?", "neg", -0.5, True, ["THEME_CONSUMABLES_AVAILABILITY", "THEME_OOS_STOCK"], []),
    ("review", "US", "en", "Flavor has completely changed! The tropical menthol used to taste great but now it's totally different. Major quality issue.", "neg", -0.7, True, ["THEME_FLAVOR_QUALITY"], []),
    ("email", "UK", "en", "Really happy with the consistency of the amber flavor sticks. Every pack tastes exactly the same - excellent quality control!", "pos", 0.9, False, ["THEME_FLAVOR_QUALITY"], []),

    # A. Product Experience - Replacement/Warranty
    ("call", "UK", "en", "I need to understand the warranty policy. My device broke and I want to know what I'm entitled to as a replacement.", "neg", -0.3, False, ["THEME_REPLACEMENT_WARRANTY"], ["authentication"]),
    ("chat", "DE", "en", "Excellent warranty service! Replacement device arrived within 2 days. The process was smooth and the agent was very helpful.", "pos", 0.9, False, ["THEME_REPLACEMENT_WARRANTY"], ["empathy", "resolution_confirmation"]),
    ("email", "US", "en", "My warranty claim has been pending for 3 weeks now with no update. This is completely unacceptable! I demand a resolution immediately.", "neg", -0.9, True, ["THEME_REPLACEMENT_WARRANTY"], ["escalation"]),

    # B. Service & Support - Contact Handling
    ("nps", "UK", "en", "The agent was incredibly empathetic and took time to understand my problem. Best customer service experience I've had!", "pos", 0.95, False, ["THEME_AGENT_EMPATHY"], ["empathy", "resolution_confirmation"]),
    ("call", "US", "en", "Agent was very rude and dismissive. Didn't listen to my issue at all. Had to repeat myself 3 times. Very poor experience.", "neg", -0.85, True, ["THEME_AGENT_EMPATHY"], []),
    ("chat", "UK", "en", "Quick resolution on the first contact! The agent understood my problem immediately and fixed it within minutes.", "pos", 0.8, False, ["THEME_FCR_RESOLUTION"], ["resolution_confirmation"]),
    ("call", "DE", "en", "I've called 4 times about the same issue and it's still not resolved. Every agent gives me different information. Extremely frustrating!", "neg", -0.9, True, ["THEME_FCR_RESOLUTION"], ["escalation"]),
    ("chat", "IT", "en", "I need to speak to your supervisor. This issue has been going on for 2 weeks and nobody is taking ownership.", "neg", -0.8, True, ["THEME_ESCALATION"], ["escalation"]),

    # B. Service & Support - SOP & Compliance
    ("call", "UK", "en", "The agent asked for my date of birth and verified my identity before proceeding - very professional and secure.", "pos", 0.5, False, ["THEME_AUTHENTICATION"], ["authentication"]),
    ("call", "UK", "en", "I've been having headaches and feeling dizzy since using the new HEETS variety. Could this be a health concern? Should I see a doctor?", "neg", -0.3, False, ["THEME_ADVERSE_EVENT"], ["adverse_event"]),
    ("chat", "US", "en", "I'm a nurse and I've noticed unusual symptoms in a patient who uses IQOS. I need to report this as a potential adverse event.", "neg", -0.4, False, ["THEME_ADVERSE_EVENT"], ["adverse_event"]),

    # C. Digital Experience - Navigation
    ("chat", "UK", "en", "Your checkout page keeps throwing an error when I try to pay. Error code PG-404. Can't complete my purchase!", "neg", -0.7, True, ["THEME_CHECKOUT_FRICTION"], []),
    ("email", "US", "en", "The search function on your website is completely broken. I can't find any products. The site is unusable!", "neg", -0.8, True, ["THEME_SEARCH_NAVIGATION"], []),
    ("nps", "DE", "en", "Website navigation is excellent and intuitive. Found exactly what I needed in less than a minute. Great UX!", "pos", 0.8, False, ["THEME_SEARCH_NAVIGATION"], []),

    # C. Digital - Account/Login
    ("chat", "UK", "en", "I can't log in to my account! It says my password is incorrect but I just reset it 5 minutes ago. Very frustrating!", "neg", -0.7, True, ["THEME_LOGIN_ACCOUNT"], []),
    ("email", "US", "en", "The chatbot gave me completely wrong information about my order status. It keeps saying the order doesn't exist when it clearly does!", "neg", -0.7, True, ["THEME_BOT_FAQ"], []),

    # D. Commerce - Delivery
    ("chat", "UK", "en", "My order was supposed to arrive 2 weeks ago but still nothing! The tracking says it's been at the local depot for 10 days.", "neg", -0.85, True, ["THEME_DELIVERY_DELAY"], []),
    ("email", "DE", "en", "Package arrived completely damaged! The device was crushed inside the box and the accessories are broken. Terrible packaging!", "neg", -0.9, True, ["THEME_DAMAGED_DELIVERY"], ["escalation"]),
    ("nps", "US", "en", "Super fast delivery! Ordered at 2pm and it arrived the next morning. Packaging was excellent and everything was perfect.", "pos", 0.95, False, ["THEME_DELIVERY_DELAY"], []),

    # D. Commerce - Payment/Order
    ("chat", "UK", "en", "I was charged twice for the same order! Please refund the duplicate charge immediately. My card statement clearly shows two charges.", "neg", -0.8, True, ["THEME_PAYMENT_REFUND"], ["escalation"]),
    ("email", "US", "en", "My discount code IQOS20 isn't working at checkout even though it says it's valid. This is so frustrating!", "neg", -0.5, True, ["THEME_PROMO_PRICING"], []),
    ("chat", "DE", "en", "I need to cancel my order immediately! I placed it by mistake and it hasn't shipped yet.", "neg", -0.2, False, ["THEME_ORDER_MANAGEMENT"], []),

    # E. Program & Loyalty
    ("chat", "UK", "en", "I just bought my first IQOS and I'm completely lost on how to set it up. The instructions in the box aren't clear at all!", "neg", -0.3, False, ["THEME_ONBOARDING"], []),
    ("email", "US", "en", "My loyalty points haven't been added to my account even though I made a purchase 2 weeks ago. I should have 500 points!", "neg", -0.5, True, ["THEME_REWARDS_LOYALTY"], []),
    ("nps", "UK", "en", "The subscription service is incredible value for money! Always reliable delivery and great pricing. Highly recommend!", "pos", 0.9, False, ["THEME_SUBSCRIPTION"], []),

    # F. Channel Experience
    ("review", "UK", "en", "In-store staff at the Oxford Street location were incredibly knowledgeable and spent time showing me how to use the device properly.", "pos", 0.85, False, ["THEME_INSTORE_SERVICE"], ["empathy"]),
    ("chat", "DE", "en", "Waited 40 minutes in the store queue just to get help with a simple question. Completely understaffed!", "neg", -0.6, True, ["THEME_INSTORE_SERVICE"], []),

    # G. Policy & Compliance
    ("call", "UK", "en", "I want to delete all my personal data from your systems immediately. I'm exercising my GDPR right to erasure.", "neg", -0.2, False, ["THEME_PRIVACY_CONCERN"], ["authentication"]),
    ("chat", "US", "en", "Do you share my data with third parties? I'm concerned about my privacy and need to understand your data practices.", "neg", -0.3, False, ["THEME_PRIVACY_CONCERN"], []),
    ("call", "UK", "en", "I want to confirm my age for the purchase. I'm 28 years old and happy to provide ID verification.", "neu", 0.0, False, ["THEME_AGE_VERIFICATION"], ["age_verification", "authentication"]),

    # NPS verbatims
    ("nps", "UK", "en", "Overall I'm a 9/10. Products are excellent but website needs improvement. Very happy with the customer service team though!", "pos", 0.7, False, [], ["resolution_confirmation"]),
    ("nps", "US", "en", "Score of 4 out of 10. Too many delivery problems and the device broke too quickly. Customer service was helpful at least.", "neg", -0.5, True, ["THEME_DELIVERY_DELAY", "THEME_DEVICE_BREAKAGE"], []),
    ("nps", "UK", "en", "10/10! Brilliant product, amazing customer service, quick delivery. You've converted me from cigarettes and I couldn't be happier!", "pos", 0.98, False, [], ["empathy"]),
    ("nps", "DE", "en", "3/10. My issues are never fully resolved on first contact. Have to keep calling back. Needs major improvement.", "neg", -0.6, True, ["THEME_FCR_RESOLUTION"], []),
]

CS_INTERACTIONS = [
    # A. Produkt - Zařízení
    ("chat", "CZ", "cs", "Moje zařízení IQOS se po 5 minutách používání přehřívá. Je to opravdu frustrující! Mám ho jen 2 měsíce.", "neg", -0.7, True, ["THEME_DEVICE_HEATING"], ["empathy"]),
    ("email", "CZ", "cs", "Baterie se vybíjí neuvěřitelně rychle - sotva vydrží 3 nabití než potřebuje nabít. Velmi zklamán kvalitou.", "neg", -0.6, True, ["THEME_BATTERY_CHARGE"], []),
    ("call", "CZ", "cs", "Moje zařízení přestalo fungovat po 6 měsících. Prostě se nezapne. Potřebuji naléhavě výměnu!", "neg", -0.8, True, ["THEME_DEVICE_BREAKAGE"], ["escalation"]),
    ("review", "CZ", "cs", "Skvělý produkt, žádné problémy s přehříváním po 3 měsících. Funguje konzistentně a výdrž baterie byla vynikající.", "pos", 0.8, False, ["THEME_DEVICE_HEATING"], []),
    ("chat", "CZ", "cs", "Moje zařízení IQOS se rozbilo během prvního týdne. Obrazovka praskla bez jakéhokoli nárazu. Velmi špatná kvalita!", "neg", -0.9, True, ["THEME_DEVICE_BREAKAGE"], ["escalation"]),

    # A. Spotřební materiál
    ("chat", "CZ", "cs", "Tabákové tyčinky HEETS jsou kompletně vyprodané ve všech blízkých obchodech i online. Kdy budou zpět naskladněny?", "neg", -0.5, True, ["THEME_CONSUMABLES_AVAILABILITY", "THEME_OOS_STOCK"], []),
    ("review", "CZ", "cs", "Chuť se úplně změnila! Tropical Menthol dříve chutnalo skvěle, ale teď je to úplně jiné. Hlavní problém s kvalitou.", "neg", -0.7, True, ["THEME_FLAVOR_QUALITY"], []),
    ("email", "CZ", "cs", "Jsem opravdu spokojený s konzistentností chuti Amber tyčinek. Každý balíček chutná naprosto stejně - výborná kontrola kvality!", "pos", 0.9, False, ["THEME_FLAVOR_QUALITY"], []),

    # B. Zákaznická podpora
    ("nps", "CZ", "cs", "Agent byl neuvěřitelně empatický a věnoval čas pochopení mého problému. Nejlepší zákaznický servis, jaký jsem kdy zažil!", "pos", 0.95, False, ["THEME_AGENT_EMPATHY"], ["empathy", "resolution_confirmation"]),
    ("call", "CZ", "cs", "Agent byl velmi hrubý a přehlíživý. Vůbec neposlouchal můj problém. Musel jsem se opakovat 3krát. Velmi špatná zkušenost.", "neg", -0.85, True, ["THEME_AGENT_EMPATHY"], []),
    ("chat", "CZ", "cs", "Rychlé vyřešení při prvním kontaktu! Agent okamžitě pochopil můj problém a vyřešil ho během minut.", "pos", 0.8, False, ["THEME_FCR_RESOLUTION"], ["resolution_confirmation"]),
    ("call", "CZ", "cs", "Volal jsem 4krát ohledně stejného problému a stále není vyřešen. Každý agent mi dává jiné informace. Extrémně frustrující!", "neg", -0.9, True, ["THEME_FCR_RESOLUTION"], ["escalation"]),

    # B. SOP & Compliance
    ("call", "CZ", "cs", "Agent si vyžádal mé datum narození a ověřil moji totožnost před pokračováním - velmi profesionální a bezpečné.", "pos", 0.5, False, ["THEME_AUTHENTICATION"], ["authentication"]),
    ("call", "CZ", "cs", "Mívám bolesti hlavy a závrať od doby, co používám novou variantu HEETS. Mohlo by to být zdravotní problém? Mám navštívit lékaře?", "neg", -0.3, False, ["THEME_ADVERSE_EVENT"], ["adverse_event"]),

    # C. Digitální zkušenost
    ("chat", "CZ", "cs", "Vaše stránka při placení neustále vrací chybu. Kód chyby PG-404. Nemohu dokončit nákup!", "neg", -0.7, True, ["THEME_CHECKOUT_FRICTION"], []),
    ("email", "CZ", "cs", "Funkce vyhledávání na vašem webu je úplně rozbitá. Nemohu najít žádné produkty. Web je nepoužitelný!", "neg", -0.8, True, ["THEME_SEARCH_NAVIGATION"], []),
    ("chat", "CZ", "cs", "Nemohu se přihlásit do svého účtu! Říká, že moje heslo je nesprávné, ale právě jsem ho resetoval před 5 minutami.", "neg", -0.7, True, ["THEME_LOGIN_ACCOUNT"], []),

    # D. Obchod & Doprava
    ("chat", "CZ", "cs", "Moje objednávka měla přijít před 2 týdny, ale stále nic! Sledování říká, že je 10 dní na místním depu.", "neg", -0.85, True, ["THEME_DELIVERY_DELAY"], []),
    ("email", "CZ", "cs", "Balíček přišel úplně poškozený! Zařízení bylo rozdrcené v krabici a příslušenství je rozbité. Hrozné balení!", "neg", -0.9, True, ["THEME_DAMAGED_DELIVERY"], ["escalation"]),
    ("chat", "CZ", "cs", "Byl mi dvakrát naúčtován stejný objednávka! Prosím okamžitě vraťte duplicitní platbu.", "neg", -0.8, True, ["THEME_PAYMENT_REFUND"], ["escalation"]),

    # E. Program & Loajalita
    ("chat", "CZ", "cs", "Právě jsem si koupil svůj první IQOS a jsem úplně ztracen v nastavení. Pokyny v krabici nejsou vůbec jasné!", "neg", -0.3, False, ["THEME_ONBOARDING"], []),
    ("email", "CZ", "cs", "Moje věrnostní body nebyly připsány na účet, přestože jsem nakoupil před 2 týdny. Měl bych mít 500 bodů!", "neg", -0.5, True, ["THEME_REWARDS_LOYALTY"], []),

    # G. Zásady & Soulad
    ("call", "CZ", "cs", "Chci smazat všechna svá osobní data z vašich systémů okamžitě. Uplatňuji právo na výmaz podle GDPR.", "neg", -0.2, False, ["THEME_PRIVACY_CONCERN"], ["authentication"]),
    ("call", "CZ", "cs", "Chci potvrdit svůj věk pro nákup. Je mi 28 let a rád poskytnu ověření totožnosti.", "neu", 0.0, False, ["THEME_AGE_VERIFICATION"], ["age_verification", "authentication"]),

    # NPS verbatims (CS)
    ("nps", "CZ", "cs", "Celkově 9 z 10. Produkty jsou vynikající, ale web potřebuje zlepšení. Velmi spokojený se zákaznickým servisem!", "pos", 0.7, False, [], ["resolution_confirmation"]),
    ("nps", "CZ", "cs", "Skóre 4 z 10. Příliš mnoho problémů s doručením a zařízení se rozbilo příliš brzy. Zákaznický servis byl alespoň nápomocný.", "neg", -0.5, True, ["THEME_DELIVERY_DELAY", "THEME_DEVICE_BREAKAGE"], []),
    ("nps", "CZ", "cs", "10/10! Skvělý produkt, úžasný zákaznický servis, rychlá doprava. Přestali jste mě kouřit cigarety a nemůžu být šťastnější!", "pos", 0.98, False, [], ["empathy"]),
]

KPI_DATA = [
    # date, market, queue, nps, csat, ces, aht_sec, repeat_rate, volume, replacements
    ("2026-01-01", "UK", "tier1", 32, 4.2, 3.8, 320, 0.12, 450, 12),
    ("2026-01-02", "UK", "tier1", 30, 4.1, 3.7, 335, 0.14, 423, 10),
    ("2026-01-03", "UK", "tier2", 28, 3.9, 3.6, 290, 0.15, 510, 15),
    ("2026-01-04", "UK", "tier1", 35, 4.3, 3.9, 310, 0.11, 490, 11),
    ("2026-01-05", "US", "tier1", 40, 4.4, 4.1, 280, 0.10, 620, 8),
    ("2026-01-06", "US", "tier1", 38, 4.3, 4.0, 295, 0.12, 580, 9),
    ("2026-01-07", "DE", "tier1", 45, 4.6, 4.3, 265, 0.08, 380, 6),
    ("2026-01-08", "CZ", "tier1", 22, 3.8, 3.5, 340, 0.18, 290, 14),
    ("2026-01-09", "CZ", "tier2", 18, 3.6, 3.3, 360, 0.20, 275, 16),
    ("2026-01-10", "UK", "tier1", 25, 3.7, 3.4, 355, 0.17, 520, 18),
    ("2026-01-11", "UK", "tier2", 20, 3.5, 3.2, 380, 0.19, 548, 22),
    ("2026-01-12", "US", "tier1", 42, 4.5, 4.2, 275, 0.09, 650, 7),
    ("2026-01-13", "DE", "tier1", 48, 4.7, 4.4, 255, 0.07, 410, 5),
    ("2026-01-14", "CZ", "tier1", 20, 3.7, 3.4, 350, 0.19, 310, 15),
    ("2026-01-15", "UK", "tier1", 33, 4.2, 3.8, 315, 0.13, 465, 11),
    ("2026-01-16", "UK", "tier1", 10, 3.2, 2.9, 420, 0.25, 670, 35),  # Anomaly spike
    ("2026-01-17", "UK", "tier1", 12, 3.3, 3.0, 410, 0.24, 720, 32),  # Anomaly spike
    ("2026-01-18", "UK", "tier1", 28, 4.0, 3.7, 330, 0.14, 500, 13),
    ("2026-01-19", "US", "tier1", 39, 4.4, 4.1, 285, 0.11, 600, 8),
    ("2026-01-20", "DE", "tier1", 44, 4.6, 4.3, 268, 0.08, 395, 6),
    ("2026-02-01", "UK", "tier1", 34, 4.2, 3.9, 318, 0.13, 480, 11),
    ("2026-02-02", "UK", "tier2", 30, 4.0, 3.7, 325, 0.15, 495, 13),
    ("2026-02-03", "US", "tier1", 41, 4.5, 4.2, 278, 0.10, 630, 8),
    ("2026-02-04", "CZ", "tier1", 24, 3.9, 3.6, 338, 0.17, 295, 13),
    ("2026-02-05", "DE", "tier1", 46, 4.7, 4.4, 260, 0.07, 405, 5),
    ("2026-02-06", "UK", "tier1", 36, 4.3, 4.0, 308, 0.12, 510, 10),
    ("2026-02-07", "UK", "tier1", 8, 3.1, 2.8, 435, 0.28, 780, 40),  # Another anomaly
    ("2026-02-08", "UK", "tier1", 31, 4.1, 3.8, 320, 0.13, 450, 12),
]

ALERTS_DATA = [
    {
        "alert_id": "ALT-0001",
        "date": "2026-01-16",
        "metric": "nps",
        "dimension_json": {"market": "UK", "queue": "tier1"},
        "direction": "down",
        "magnitude": 3.2,
        "alert_type": "kpi_drop",
        "z_score": 3.2,
        "status": "open",
        "explanation": "NPS drop: current=10.0, baseline=31.5, change=68.3%, z-score=3.20",
        "playbook_en": "KPI metric drop. Actions: 1) Root cause analysis, 2) Service quality review, 3) Operational improvements",
        "playbook_cs": "Pokles KPI metriky. Akce: 1) Analýza příčin, 2) Přezkoumání kvality, 3) Operativní zlepšení",
        "notes": "Correlated with delivery delay spike",
    },
    {
        "alert_id": "ALT-0002",
        "date": "2026-01-17",
        "metric": "volume",
        "dimension_json": {"market": "UK"},
        "direction": "up",
        "magnitude": 2.8,
        "alert_type": "volume_spike",
        "z_score": 2.8,
        "status": "open",
        "explanation": "volume spike: current=720, baseline=487.2, change=47.8%, z-score=2.80",
        "playbook_en": "High volume spike detected. Consider: 1) Check for product/service incident, 2) Increase staffing, 3) Prepare FAQ/bot responses",
        "playbook_cs": "Detekován spike objemu. Zvažte: 1) Zkontrolujte produkt/službu, 2) Zvyšte obsazení, 3) Připravte odpovědi FAQ/bota",
        "notes": "",
    },
    {
        "alert_id": "ALT-0003",
        "date": "2026-02-07",
        "metric": "nps",
        "dimension_json": {"market": "UK"},
        "direction": "down",
        "magnitude": 3.5,
        "alert_type": "kpi_drop",
        "z_score": 3.5,
        "status": "open",
        "explanation": "NPS drop: current=8.0, baseline=32.4, change=75.3%, z-score=3.50",
        "playbook_en": "KPI metric drop. Actions: 1) Root cause analysis, 2) Service quality review, 3) Operational improvements",
        "playbook_cs": "Pokles KPI metriky. Akce: 1) Analýza příčin, 2) Přezkoumání kvality, 3) Operativní zlepšení",
        "notes": "Delivery delay issues in UK market",
    },
]


def generate_interactions():
    """Generate full interaction dataset."""
    all_interactions = []

    def make_id(i):
        return f"IID-{i:06d}"

    idx = 1
    base_date = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for channel, market, lang, text, sentiment, score, dsat, topics, behaviors in EN_INTERACTIONS:
        days_offset = random.randint(0, 60)
        hours_offset = random.randint(0, 23)
        ts = base_date + timedelta(days=days_offset, hours=hours_offset)

        all_interactions.append({
            "interaction_id": make_id(idx),
            "channel": channel,
            "source_system": "demo",
            "created_ts": ts.isoformat(),
            "lang": lang,
            "market": market,
            "author": f"customer_{idx:05d}",
            "raw_text": text,
            "audio_path": "",
            "pii_masked_text": text,
            "meta_json": json.dumps({
                "topics": topics,
                "nps_score": random.randint(0, 10) if channel == "nps" else None,
                "queue": random.choice(["tier1", "tier2"]),
                "vendor": random.choice(["vendorA", "vendorB", "vendorC"]),
            }),
        })
        idx += 1

    for channel, market, lang, text, sentiment, score, dsat, topics, behaviors in CS_INTERACTIONS:
        days_offset = random.randint(0, 60)
        hours_offset = random.randint(0, 23)
        ts = base_date + timedelta(days=days_offset, hours=hours_offset)

        all_interactions.append({
            "interaction_id": make_id(idx),
            "channel": channel,
            "source_system": "demo",
            "created_ts": ts.isoformat(),
            "lang": lang,
            "market": market,
            "author": f"customer_{idx:05d}",
            "raw_text": text,
            "audio_path": "",
            "pii_masked_text": text,
            "meta_json": json.dumps({
                "topics": topics,
                "nps_score": random.randint(0, 10) if channel == "nps" else None,
                "queue": random.choice(["tier1", "tier2"]),
                "vendor": random.choice(["vendorA", "vendorB", "vendorC"]),
            }),
        })
        idx += 1

    return all_interactions


def write_csv(data, filename, fieldnames):
    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Written {len(data)} rows to {filename}")


if __name__ == "__main__":
    import os
    os.makedirs(".", exist_ok=True)

    interactions = generate_interactions()
    interaction_fields = [
        "interaction_id", "channel", "source_system", "created_ts",
        "lang", "market", "author", "raw_text", "audio_path",
        "pii_masked_text", "meta_json",
    ]
    write_csv(interactions, "interactions_sample.csv", interaction_fields)

    kpi_fields = ["date", "market", "queue", "nps", "csat", "ces", "aht_sec", "repeat_rate", "volume", "replacements"]
    write_csv(
        [dict(zip(kpi_fields, row)) for row in KPI_DATA],
        "kpi_daily_sample.csv",
        kpi_fields,
    )

    with open("alerts_sample.json", "w", encoding="utf-8") as f:
        json.dump(ALERTS_DATA, f, indent=2)
    print(f"Written {len(ALERTS_DATA)} alerts to alerts_sample.json")

    print("Sample data generation complete!")
