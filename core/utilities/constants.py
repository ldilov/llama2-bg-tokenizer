ALPHABET = [
    r'а', r'б', r'в', r'г', r'д', r'е', r'ж', r'з', r'и', r'й', r'к', r'л',
    r'м', r'н', r'о', r'п', r'р', r'с', r'т', r'у', r'ф', r'х', r'ц', r'ч',
    r'ш', r'щ', r'ь', r'ю', r'я', r'0', r'1', r'2', r'3', r'4', r'5', r'6',
    r'7', r'8', r'9', r',', r'\.', r'!', r'?', r'ъ', r'Ъ',
    r';', r':', r'А', r'Б', r'В', r'Г', r'Д', r'Е', r'Ж', r'З', r'И',
    r'Й', r'К', r'Л', r'М', r'Н', r'О', r'П', r'Р', r'С', r'Т', r'У', r'Ф',
    r'Х', r'Ц', r'Ч', r'Ш', r'Щ', r'ѝ', r'Ю', r'Я', '\\', r'/', r'"', r'^',
    r'`', r'+', '\-', r'*', r'%', r'#', r'$', r'~', r'=', r'&', r'|', '\''
]

CHINESE_CHARACTER_RANGES = [
    r'\u9FFD-\u9FFF',
]

OTHER_UNK_CHARACTERS = [
    r'є', r'ѕ', r'і', r'ј', r'љ', r'њ', r'ћ', r'ѝ', r'џ',
    r'ѡ', r'ѣ', r'ѧ', r'ѫ', r'ѭ', r'һ', r'ա', r'ն', r'ր',
    r'א', r'ב', r'ג', r'ד', r'ה', r'ו', r'ח', r'ט', r'י',
    r'כ', r'ל', r'ם', r'מ', r'ן', r'נ', r'ס', r'ע', r'פ',
    r'צ', r'ק', r'ר', r'ש', r'ת', r'،', r'ا', r'ب', r'ة',
    r'ت', r'ث', r'ج', r'ح', r'خ', r'د', r'ذ', r'ر', r'ز',
    r'س', r'ش', r'ص', r'ض', r'ط', r'ع', r'غ', r'ـ', r'ف',
    r'ق', r'ك', r'ل', r'م', r'ن', r'ه', r'و', r'ى', r'ي',
    r'َ', r'ُ', r'ِ', r'ّ', r'ْ', r'ٓ', r'ٔ', r'ٕ', r'پ',
    r'چ', r'ک', r'گ', r'ی', r'݀', r'݁', r'ं', r'क', r'ग',
    r'ज', r'त', r'द', r'न', r'प', r'म', r'य', r'र', r'ल',
    r'व', r'स', r'ह', r'ा', r'ि', r'ी', r'ु', r'े', r'ो',
    r'्', r'।', r'ক', r'ন', r'র', r'া', r'ি', r'ে', r'্',
    r'ก', r'ค', r'ง', r'ด', r'ต', r'ท', r'น', r'บ', r'ป',
    r'ม', r'ย', r'ร', r'ล', r'ว', r'ส', r'ห', r'อ', r'ะ',
    r'ั', r'า', r'ิ', r'ี', r'เ', r'่', r'้', r'ა', r'ი',
    r'ᄀ', r'ᄂ', r'ᄃ', r'ᄅ', r'ᄆ', r'ᄇ', r'ᄉ', r'ᄋ', r'ᄌ',
    r'ᄎ', r'ᄐ', r'ᄑ', r'ᄒ', r'ᅡ', r'ᅢ', r'ᅥ', r'ᅦ',
    r'ᅧ', r'ᅩ', r'ᅪ', r'ᅮ', r'ᅳ', r'ᅵ', r'ᆨ', r'ᆫ', r'ᆯ',
    r'ᆷ', r'ᆸ', r'ᆼ',
    r'†', r'•',
    r'■',
    r'▢', r'▪', r'△', r'►', r'◊', r'●', r'★', r'☺', r'♠',
    r'♤', r'♥', r'♦', r'✅', r'✈', r'✓', r'✔', r'❑', r'❖',
    r'❤', r'➡', r'➤', r'、', r'。', r'「', r'」', r'【', r'】',
    r'あ', r'い', r'う', r'え', r'お', r'か', r'き', r'く', r'け',
    r'こ', r'さ', r'し', r'す', r'せ', r'そ', r'た', r'ち', r'っ',
    r'つ', r'て', r'と', r'な', r'に', r'の', r'は', r'ま', r'み',
    r'め', r'も', r'や', r'よ', r'ら', r'り', r'る', r'れ', r'わ',
    r'を', r'ん', r'゙', r'゚', r'ア', r'ィ', r'イ', r'ウ', r'ェ',
    r'エ', r'オ', r'カ', r'キ', r'ク', r'ケ', r'コ', r'サ', r'シ',
    r'ス', r'セ', r'タ', r'チ', r'ッ', r'ツ', r'テ', r'ト', r'ナ',
    r'ニ', r'ハ', r'ヒ', r'フ', r'ヘ', r'ホ', r'マ', r'ミ', r'ム',
    r'メ', r'ャ', r'ュ', r'ョ', r'ラ', r'リ', r'ル', r'レ', r'ロ',
    r'ン', r'・', r'ー'
]