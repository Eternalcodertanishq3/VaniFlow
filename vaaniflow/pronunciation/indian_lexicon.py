"""
Curated Indian name/place/brand pronunciation dictionary.
Maps common text -> phonetic hint that TTS engines handle correctly.

Format: "written form" -> "phonetic hint for TTS"
These are NOT IPA — they're written hints that TTS reads naturally.
"""

INDIAN_PRONUNCIATION_MAP = {
    # Cities
    "Bangalore":    "Baanga-lore",
    "Bengaluru":    "Benga-loo-roo",
    "Mumbai":       "Moom-bye",
    "Chennai":      "Chen-eye",
    "Hyderabad":    "High-dra-baad",
    "Ahmedabad":    "Ahm-eh-da-baad",
    "Pune":         "Poo-neh",
    "Kolkata":      "Kol-kaa-ta",
    "Thiruvananthapuram": "Thi-roo-va-nan-tha-poo-ram",
    "Bhubaneswar":  "Bhu-ba-nesh-war",
    "Visakhapatnam":"Vi-shaak-ha-pat-nam",
    "Coimbatore":   "Koyam-ba-tore",

    # People (cricketers, politicians, actors)
    "Virat Kohli":      "Vee-rat Koh-lee",
    "Sachin Tendulkar": "Saa-chin Ten-dool-kar",
    "Narendra Modi":    "Na-ren-dra Mo-dee",
    "Amitabh Bachchan": "A-mee-taab Bach-chan",
    "Priyanka Chopra":  "Pri-yan-ka Cho-pra",
    "Deepika Padukone": "Deep-ika Pa-doo-ko-neh",

    # Brands/Companies
    "Infosys":      "Info-sis",
    "Wipro":        "Wip-ro",
    "Zomato":       "Zo-maa-to",
    "Swiggy":       "Swig-ee",
    "Paytm":        "Pay-tee-em",
    "Jio":          "Jee-oh",
    "BYJU'S":       "Bye-joos",
    "Ola":          "Oh-la",
    "Nykaa":        "Ny-kaa",

    # Common Hindi/Sanskrit words in English text
    "namaste":      "na-mas-teh",
    "chai":         "chye",
    "yoga":         "yo-ga",
    "karma":        "kar-ma",
    "nirvana":      "nir-vaa-na",
    "chakra":       "chak-ra",
    "guru":         "goo-roo",
    "mantra":       "man-tra",

    # Everyday Indian Terms
    "Aadhar":       "Aa-dhaar",
    "Aadhaar":      "Aa-dhaar",
    "crore":        "krore",
    "lakh":         "laakh",
    "rupees":       "roo-peez",
    "lakhs":        "laakhs",
    "crores":       "krores",

    # Additional Cities
    "Kanpur":       "Kaan-poor",
    "Lucknow":      "Lak-now",
    "Gurugram":     "Goo-roo-graam",
    "Noida":        "Noy-da",
    "Indore":       "In-dore",
    "Bhopal":       "Bho-paal",

    # States
    "Maharashtra":  "Ma-ha-raash-tra",
    "Tamil Nadu":   "Ta-mil Naa-doo",
    "Karnataka":    "Kar-naa-ta-ka",
    "Telangana":    "Te-lan-gaa-na",
    "Rajasthan":    "Raa-jas-thaan",
    "Uttarakhand":  "Ut-ta-ra-khand",
}
