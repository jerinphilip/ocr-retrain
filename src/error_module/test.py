from dictionary import Dictionary

D = Dictionary(lang='malayalam')

samples = [
	"സി",
	"എം",
	"തന്നെ",
	"കെ",
	"പുതിയ",
	"എ",
	"എസ്‌",
	"ബി",
	"ഐ",
	"കഴിഞ്ഞ",
	"ആ",
	"അദ്ദേഹം",
	"എന്നാല്‍",
	"നിന്ന്",
	"രൂപ",
	"ടി"
]

for sample in samples:
	print(sample, D.error(sample))
