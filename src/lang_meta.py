"""Per-language covariates: script, Joshi et al. (2020) resource class, Wikipedia article count.
Builds data/lang_meta.csv. Romanised duplicates (xxx_Latn of a non-Latin language) share the
language's resource class but differ in script -> within-language script contrast."""
import csv, os, sys
sys.path.insert(0, os.path.dirname(__file__))

# code: (name, joshi lookup name, wikipedia prefix or None, is_romanised_variant)
L = {
 "acm_Arab": ("Mesopotamian Arabic", "mesopotamian arabic", None, 0),
 "afr_Latn": ("Afrikaans", "afrikaans", "af", 0),
 "als_Latn": ("Tosk Albanian", "albanian", "sq", 0),
 "amh_Ethi": ("Amharic", "amharic", "am", 0),
 "apc_Arab": ("North Levantine Arabic", "north levantine arabic", None, 0),
 "arb_Arab": ("Standard Arabic", "arabic", "ar", 0),
 "arb_Latn": ("Arabic (romanised)", "arabic", "ar", 1),
 "ars_Arab": ("Najdi Arabic", "najdi arabic", None, 0),
 "ary_Arab": ("Moroccan Arabic", "moroccan arabic", "ary", 0),
 "arz_Arab": ("Egyptian Arabic", "egyptian arabic", "arz", 0),
 "asm_Beng": ("Assamese", "assamese", "as", 0),
 "azj_Latn": ("North Azerbaijani", "azerbaijani", "az", 0),
 "bam_Latn": ("Bambara", "bambara", "bm", 0),
 "ben_Beng": ("Bengali", "bengali", "bn", 0),
 "ben_Latn": ("Bengali (romanised)", "bengali", "bn", 1),
 "bod_Tibt": ("Tibetan", "tibetan", "bo", 0),
 "bul_Cyrl": ("Bulgarian", "bulgarian", "bg", 0),
 "cat_Latn": ("Catalan", "catalan", "ca", 0),
 "ceb_Latn": ("Cebuano", "cebuano", "ceb", 0),
 "ces_Latn": ("Czech", "czech", "cs", 0),
 "ckb_Arab": ("Central Kurdish", "kurdish (sorani)", "ckb", 0),
 "dan_Latn": ("Danish", "danish", "da", 0),
 "deu_Latn": ("German", "german", "de", 0),
 "ell_Grek": ("Greek", "greek", "el", 0),
 "eng_Latn": ("English", "english", "en", 0),
 "est_Latn": ("Estonian", "estonian", "et", 0),
 "eus_Latn": ("Basque", "basque", "eu", 0),
 "fin_Latn": ("Finnish", "finnish", "fi", 0),
 "fra_Latn": ("French", "french", "fr", 0),
 "fuv_Latn": ("Nigerian Fulfulde", "fulfulde", "ff", 0),
 "gaz_Latn": ("West Central Oromo", "oromo", "om", 0),
 "grn_Latn": ("Guarani", "guarani", "gn", 0),
 "guj_Gujr": ("Gujarati", "gujarati", "gu", 0),
 "hat_Latn": ("Haitian Creole", "haitian", "ht", 0),
 "hau_Latn": ("Hausa", "hausa", "ha", 0),
 "heb_Hebr": ("Hebrew", "hebrew", "he", 0),
 "hin_Deva": ("Hindi", "hindi", "hi", 0),
 "hin_Latn": ("Hindi (romanised)", "hindi", "hi", 1),
 "hrv_Latn": ("Croatian", "croatian", "hr", 0),
 "hun_Latn": ("Hungarian", "hungarian", "hu", 0),
 "hye_Armn": ("Armenian", "armenian", "hy", 0),
 "ibo_Latn": ("Igbo", "igbo", "ig", 0),
 "ilo_Latn": ("Ilocano", "ilocano", "ilo", 0),
 "ind_Latn": ("Indonesian", "indonesian", "id", 0),
 "isl_Latn": ("Icelandic", "icelandic", "is", 0),
 "ita_Latn": ("Italian", "italian", "it", 0),
 "jav_Latn": ("Javanese", "javanese", "jv", 0),
 "jpn_Jpan": ("Japanese", "japanese", "ja", 0),
 "kac_Latn": ("Jingpho", "jingpho", None, 0),
 "kan_Knda": ("Kannada", "kannada", "kn", 0),
 "kat_Geor": ("Georgian", "georgian", "ka", 0),
 "kaz_Cyrl": ("Kazakh", "kazakh", "kk", 0),
 "kea_Latn": ("Kabuverdianu", "kabuverdianu", None, 0),
 "khk_Cyrl": ("Halh Mongolian", "mongolian", "mn", 0),
 "khm_Khmr": ("Khmer", "khmer", "km", 0),
 "kin_Latn": ("Kinyarwanda", "kinyarwanda", "rw", 0),
 "kir_Cyrl": ("Kyrgyz", "kirghiz", "ky", 0),
 "kor_Hang": ("Korean", "korean", "ko", 0),
 "lao_Laoo": ("Lao", "lao", "lo", 0),
 "lin_Latn": ("Lingala", "lingala", "ln", 0),
 "lit_Latn": ("Lithuanian", "lithuanian", "lt", 0),
 "lug_Latn": ("Ganda", "luganda", "lg", 0),
 "luo_Latn": ("Luo", "luo", None, 0),
 "lvs_Latn": ("Standard Latvian", "latvian", "lv", 0),
 "mal_Mlym": ("Malayalam", "malayalam", "ml", 0),
 "mar_Deva": ("Marathi", "marathi", "mr", 0),
 "mkd_Cyrl": ("Macedonian", "macedonian", "mk", 0),
 "mlt_Latn": ("Maltese", "maltese", "mt", 0),
 "mri_Latn": ("Maori", "maori", "mi", 0),
 "mya_Mymr": ("Burmese", "burmese", "my", 0),
 "nld_Latn": ("Dutch", "dutch", "nl", 0),
 "nob_Latn": ("Norwegian Bokmal", "norwegian (bokmål)", "no", 0),
 "npi_Deva": ("Nepali", "nepali", "ne", 0),
 "npi_Latn": ("Nepali (romanised)", "nepali", "ne", 1),
 "nso_Latn": ("Northern Sotho", "northern sotho", "nso", 0),
 "nya_Latn": ("Nyanja", "chichewa", "ny", 0),
 "ory_Orya": ("Odia", "odia", "or", 0),
 "pan_Guru": ("Eastern Panjabi", "panjabi", "pa", 0),
 "pbt_Arab": ("Southern Pashto", "pashto", "ps", 0),
 "pes_Arab": ("Western Persian", "persian", "fa", 0),
 "plt_Latn": ("Plateau Malagasy", "malagasy", "mg", 0),
 "pol_Latn": ("Polish", "polish", "pl", 0),
 "por_Latn": ("Portuguese", "portuguese", "pt", 0),
 "ron_Latn": ("Romanian", "romanian", "ro", 0),
 "rus_Cyrl": ("Russian", "russian", "ru", 0),
 "shn_Mymr": ("Shan", "shan", "shn", 0),
 "sin_Latn": ("Sinhala (romanised)", "sinhala", "si", 1),
 "sin_Sinh": ("Sinhala", "sinhala", "si", 0),
 "slk_Latn": ("Slovak", "slovak", "sk", 0),
 "slv_Latn": ("Slovenian", "slovenian", "sl", 0),
 "sna_Latn": ("Shona", "shona", "sn", 0),
 "snd_Arab": ("Sindhi", "sindhi", "sd", 0),
 "som_Latn": ("Somali", "somali", "so", 0),
 "sot_Latn": ("Southern Sotho", "sesotho", "st", 0),
 "spa_Latn": ("Spanish", "spanish", "es", 0),
 "srp_Cyrl": ("Serbian", "serbian", "sr", 0),
 "ssw_Latn": ("Swati", "swati", "ss", 0),
 "sun_Latn": ("Sundanese", "sundanese", "su", 0),
 "swe_Latn": ("Swedish", "swedish", "sv", 0),
 "swh_Latn": ("Swahili", "swahili", "sw", 0),
 "tam_Taml": ("Tamil", "tamil", "ta", 0),
 "tel_Telu": ("Telugu", "telugu", "te", 0),
 "tgk_Cyrl": ("Tajik", "tajik", "tg", 0),
 "tgl_Latn": ("Tagalog", "tagalog", "tl", 0),
 "tha_Thai": ("Thai", "thai", "th", 0),
 "tir_Ethi": ("Tigrinya", "tigrinya", "ti", 0),
 "tsn_Latn": ("Tswana", "tswana", "tn", 0),
 "tso_Latn": ("Tsonga", "tsonga", "ts", 0),
 "tur_Latn": ("Turkish", "turkish", "tr", 0),
 "ukr_Cyrl": ("Ukrainian", "ukrainian", "uk", 0),
 "urd_Arab": ("Urdu", "urdu", "ur", 0),
 "urd_Latn": ("Urdu (romanised)", "urdu", "ur", 1),
 "uzn_Latn": ("Northern Uzbek", "uzbek", "uz", 0),
 "vie_Latn": ("Vietnamese", "vietnamese", "vi", 0),
 "war_Latn": ("Waray", "waray", "war", 0),
 "wol_Latn": ("Wolof", "wolof", "wo", 0),
 "xho_Latn": ("Xhosa", "xhosa", "xh", 0),
 "yor_Latn": ("Yoruba", "yoruba", "yo", 0),
 "zho_Hans": ("Chinese (Simplified)", "mandarin", "zh", 0),
 "zho_Hant": ("Chinese (Traditional)", "mandarin", "zh", 0),
 "zsm_Latn": ("Standard Malay", "malay", "ms", 0),
 "zul_Latn": ("Zulu", "zulu", "zu", 0),
}


def build():
    from data import all_languages
    langs = all_languages()
    missing = set(langs) - set(L); extra = set(L) - set(langs)
    assert not missing and not extra, (missing, extra)
    joshi = {}
    with open("data/lang2tax.txt") as f:
        for line in f:
            if "," in line:
                n, c = line.rstrip("\n").rsplit(",", 1)
                joshi[n.strip().lower()] = int(c)
    wiki = {}
    with open("data/wikistats_raw.csv") as f:
        for r in csv.DictReader(f):
            if r.get("type") == "wikipedia" and r.get("prefix") and (r.get("good") or "").isdigit():
                wiki[r["prefix"]] = int(r["good"])
    rows, nojoshi = [], []
    for code in langs:
        name, jn, wp, rom = L[code]
        jc = joshi.get(jn)
        if jc is None:
            nojoshi.append((code, jn))
        rows.append(dict(lang=code, name=name, script=code.split("_")[1],
                         latin=int(code.endswith("_Latn")), romanised=rom,
                         joshi_class=jc, wiki_articles=wiki.get(wp) if wp else None))
    os.makedirs("data", exist_ok=True)
    with open("data/lang_meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("wrote data/lang_meta.csv", len(rows)); print("no joshi class:", nojoshi)
    from collections import Counter
    print("joshi class counts", sorted(Counter(r["joshi_class"] for r in rows).items(), key=str))
    print("script counts", Counter(r["script"] for r in rows).most_common())


if __name__ == "__main__":
    build()
