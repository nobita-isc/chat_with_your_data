from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """
父親を橋の上から投げ落として殺害したとして、大阪府警は11日、無職の玉井将太容疑者（36）＝大阪市淀川区新高5＝を殺人の疑いで逮捕したと発表した。父重弘さん（65）は10日未明、自宅近くの三国橋から転落死していた。玉井容疑者は「仕事ができず、嫌気が差して父親を道連れに自殺しようと考えた」と容疑を認めているという。

　逮捕容疑は10日午前1時ごろ、淀川区西三国4の三国橋で重弘さんの両足をつかみ、欄干（高さ約1・2メートル）を乗り越えさせ、下を流れる神崎川に投げ落として殺害したとしている。

　捜査1課によると、玉井容疑者は両親と弟、妹との5人暮らし。母親が10日午前7時45分ごろ、「息子が橋の上から父親を投げたと言っている」と110番していた。玉井容疑者は父親の夕食に睡眠薬を混ぜてレンタカーで連れ出したといい、「自分も別の場所で川に飛び込んだが死にきれなかった」などと供述しているという。【郡悠介】
"""
print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))
[{'summary_text': 'Liana Barrientos, 39, is charged with two counts of "offering a false instrument for filing in the first degree" In total, she has been married 10 times, with nine of her marriages occurring between 1999 and 2002. She is believed to still be married to four men.'}]