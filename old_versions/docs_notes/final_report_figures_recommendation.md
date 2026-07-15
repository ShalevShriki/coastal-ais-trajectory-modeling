# המלצה סופית — גרפים לדוח הסופי

## גרפים שחייבים להיות בדוח

### 1. גרף סריקת אורך ההיסטוריה

**הגרף:** Fixed-context AR LSTM — history length sweep  
**מיקום:** סעיף 6.2 — סריקת אורך היסטוריה ב־AR LSTM  
**קובץ:** `fig_ar_context_sweep.png`

**למה הוא חיוני:** זה הגרף שעונה באופן הישיר ביותר על שאלת המחקר של הפרויקט:

- 9h, 12h, 18h, 24h

והוא מציג בצורה ברורה:

```text
20.43 → 19.99 → 19.71 → 20.30
```

כלומר:

- מעבר מ־9 ל־18 שעות משפר
- 18 שעות הן הטובות ביותר ב־AR
- מעבר ל־24 שעות מחמיר
- לכן **יותר היסטוריה אינה תמיד טובה יותר**

#### שינוי חשוב לפני ההכנסה

מומלץ **להסיר מהגרף הראשי** את הנקודה הכתומה:

```text
AR 12h no land penalty = 19.59
```

היא אינה שייכת לסריקת ההיסטוריה הנקייה, כי שיניתם בה גם את אורך ההיסטוריה וגם את תנאי ה־loss.

בגרף הראשי כל הנקודות צריכות להיות:

$$
\lambda_{\mathrm{land}} = 0.1
$$

את נקודת ה־no-land אפשר:

- להזכיר במשפט
- להעביר ל־ablation קטן
- או להכניס לנספח

אחרת הקורא עלול לחשוב ש־12 שעות הן למעשה הטובות ביותר, אף שהתוצאה נובעת גם מהסרת קנס היבשה.

#### פסקה מתאימה מתחת לגרף

> The history-length sweep shows a non-monotonic relationship between temporal context and forecasting accuracy. Increasing the context from 9h to 18h improves median FDE, but extending it to 24h degrades performance. This suggests that additional history may contain outdated motion information or become difficult for the recurrent encoder to compress effectively.

---

### 2. גרף Straight מול Maneuver לפי אורך היסטוריה

**הגרף:** AR LSTM — straight vs maneuver by history length  
**מיקום:** סעיף 6.3 — ניתוח לפי סוג תנועה  
**קובץ:** `fig_ar_straight_vs_maneuver.png`

**למה הוא חיוני:** הוא מסביר למה בכלל היה הגיוני לבנות מודל אדפטיבי.

הוא מראה:

- עבור **Straight**, ‏18 שעות הן הטובות ביותר:  
  `48.3, 44.2, 39.3, 43.2`
- עבור **Maneuver**, ‏9 שעות הן הטובות ביותר:  
  `16.7, 17.0, 18.4, 19.3`

זה הפאנץ׳ליין שמוביל טבעית ל־Adaptive Model:

> אם חלון קצר טוב יותר לסוג אחד וחלון ארוך טוב יותר לסוג אחר, אולי המודל יכול ללמוד לבחור לבד.

#### בעיה שצריך לתקן

בכיתוב של הגרף כתוב:

> Bucket sizes differ slightly by history length.

זה מחליש את המסקנה, כי ייתכן שהעמודות אינן מחושבות בדיוק על אותן דוגמאות.

לדוגמה, אם הגדרת Straight/Maneuver מחושבת מחדש על כל חלון:

- מסלול יכול להיחשב Straight ב־9 שעות
- אבל Maneuver ב־24 שעות

ואז לא משווים את אותם buckets בין ארבעת המודלים.

#### לניסוי נקי עדיף

1. להגדיר Straight/Maneuver **פעם אחת** לפי קריטריון משותף
2. להשתמש באותה רשימת test samples לכל ארבעת המודלים
3. להשוות כל מודל על אותן דוגמאות בדיוק

למשל, לקבוע את סוג התנועה לפי 24 שעות ההיסטוריה המלאות או לפי העתיד האמיתי, ואז לא לשנות את הסיווג בין המודלים.

אם לא ניתן לחשב מחדש, צריך להוסיף הסתייגות ברורה:

> Bucket membership differs slightly across context lengths; therefore, the result indicates a trend rather than a perfectly controlled within-sample comparison.

הגרף חשוב, אבל עדיף לתקן אותו לפני שהופכים אותו לבסיס מרכזי של המסקנה.

---

### 3. גרף התנהגות השער האדפטיבי

**הגרף:** Adaptive multi-scale gate behavior  
**מיקום:** סעיף 6.4 — תוצאת המודל האדפטיבי  
**קובץ:** `fig_adaptive_alphas.png`

**למה הוא חיוני:** זה הגרף המרכזי של הפאנץ׳ליין האדפטיבי.

החלק השמאלי מראה את המשקל הרך הממוצע:

$$
\bar{\alpha}_9 = 0.146,\quad
\bar{\alpha}_{12} = 0.224,\quad
\bar{\alpha}_{18} = 0.289,\quad
\bar{\alpha}_{24} = 0.341
$$

אבל החלק הימני חשוב אפילו יותר:

$$
\operatorname{argmax}(\alpha)=24\mathrm{h}
\quad\text{ב־}86.1\%\text{ מהדוגמאות.}
$$

כך הקורא מבין מיד:

- האלפות אינן קבועות לחלוטין
- קיימת תערובת רכה
- אבל בפועל השער כמעט תמיד נותן את המקום הראשון ל־24 שעות
- לכן הוא לא למד מעבר חד ומשמעותי בין קצר לארוך

זה הרבה יותר חזק מטבלה בלבד.

#### פסקה מתאימה

> Although all four contexts receive non-zero soft weights, the gate is strongly biased toward the longest history: 24h is the highest-weight context for 86.1% of the coastal test samples. Therefore, the gate does not exhibit the expected specialization in which maneuvering trajectories select short context and stable trajectories select long context.

ה־Random Forest וה־Spearman עדיין צריכים להופיע בטקסט או בטבלה קטנה. הגרף הזה לא מוכיח לבדו את ה־geographic prior; הוא מוכיח בעיקר את הדומיננטיות של 24 שעות.

---

### 4. חסר לכם גרף חשוב יותר מכמה מהגרפים הקיימים

בסעיף 6.1 אתם כותבים שיש “טבלה ואיור” של הדירוג הכללי, אבל צריך **גרף נקי** של דירוג כל המודלים.

זה צריך להיות **איור 1 של פרק התוצאות**.

הגרף צריך להציג:

| Model | Median FDE |
|-------|------------|
| Kinematic baseline | 102.5 |
| Flat LSTM | 18.80 |
| Transformer | 19.40 |
| AR 18h | 19.71 |
| AR 12h | 19.99 |
| Adaptive | 20.19 |
| AR 24h | 20.30 |
| AR 9h | 20.43 |
| Sliding | 22.44 |

**קובץ קיים:** `fig_model_ranking_fde.png` (מומלץ לשפר — ראו למטה)

לא כדאי לשים את כולם על ציר רגיל אחד, כי 102.5 ימחץ את ההבדלים בין 18.8 ל־22.4.

אפשר לבחור אחת משתי אפשרויות:

1. **שני panels:** בייסליין מול מודלים נוירוניים, ואז zoom על המודלים הנוירוניים
2. **גרף של המודלים הנוירוניים בלבד**, עם הערה בולטת:  
   `Kinematic baseline = 102.5 km`

זה גרף יותר חשוב מהגרף שמשווה Straight/Maneuver בין כל הארכיטקטורות.

---

## גרפים שמתאימים, אבל רק בתנאים מסוימים

### 5. גרף התפתחות השגיאה לאורך 12 שעות

**הגרף:** Coastal suite — error growth over the 12h horizon  
**מיקום אפשרי:** סוף סעיף 6.1 או בפרק 7 — דיון  
**קובץ:** `fig_error_vs_horizon.png`

**מה הוא תורם:** הוא מראה שהשגיאה גדלה עם אופק החיזוי:

$$
d(t)\uparrow \quad \text{כאשר} \quad t\uparrow
$$

והוא יכול לתמוך בדיון על:

- הצטברות שגיאה אוטורגרסיבית
- ההבדל בין ADE ל־FDE
- היתרון האפשרי של Flat LSTM
- העובדה שהמודלים קרובים בתחילת האופק ונפרדים יותר בהמשך

#### הבעיה הנוכחית

בכותרת כתוב:

> median over saved test samples

אם מדובר רק במספר קטן של דוגמאות שנשמרו לציור, לא נכון להשתמש בגרף כדי לטעון טענה כללית על כלל ה־test set.

גם רואים שסדר המודלים בנקודת 12 שעות אינו זהה בדיוק לדירוג ה־median FDE המלא. זה כנראה נובע מכך שמדובר ב־subset.

לכן:

- אם ניתן לחשב מחדש על **כל** test set — הגרף מתאים מאוד לדוח
- אם הוא מבוסס רק על saved samples — להעביר לנספח או להסיר

**הכותרת הרצויה:**

> Median geodesic error over the full coastal test set

ולא:

> over saved test samples

---

### 6. דוגמאות מסלול איכותניות

**הגרף:** AR LSTM 12h — שישה מסלולים על מפה  
**מיקום אפשרי:** סוף פרק 6 או בפרק 7 — דיון וכשלי המודל  
**קובץ:** `fig_tracks_report_6panel.png`

**מה הוא תורם:** הוא הופך את FDE ממשהו מספרי למשהו שהקורא יכול להבין חזותית:

- תחזית טובה: FDE = 2.2 ק"מ
- תחזית בינונית: FDE = 14.9 ק"מ
- כשל משמעותי: FDE = 102.3 או 234.5 ק"מ

הוא גם מראה:

- כיצד טעות בכיוון יוצרת FDE גדול
- כיצד מסלול חזוי יכול להגיע לאזור שונה לחלוטין
- למה median לבדו אינו מספר את כל הסיפור
- למה יש צורך לדווח על tail failures

#### לא הייתי שם אותו כפי שהוא בגוף הדוח

שישה panels תופסים הרבה מקום, וחלקם אינם קשורים ישירות לשאלת אורך ההיסטוריה.

עדיף לצמצם ל**שלוש** דוגמאות:

1. דוגמה טובה, סביב percentile נמוך
2. דוגמה טיפוסית, סביב median
3. דוגמת כשל, למשל percentile ‏95 או 99

ולכתוב במפורש איך נבחרו, כדי שלא ייראה כמו cherry-picking:

> Examples were selected near the 10th, 50th, and 95th percentiles of the test FDE distribution.

את הגרסה המלאה עם שישה מסלולים אפשר לשים בנספח.

---

## גרפים שלא כדאי להכניס לגוף הדוח

### 7. שני גרפי ה־Training History

יש לכם שתי גרסאות כמעט זהות:

- גרסה עם שני panels (`fig_train_val_loss_ar12h.png`)
- גרסה עם teacher forcing על הציר הימני בתוך הגרף העליון (`fig_train_val_comparable_with_tf.png`)

**אסור להכניס את שתיהן.** הן מספרות כמעט אותו סיפור.

#### האם להכניס אחת מהן?

הן מסבירות היטב:

- teacher forcing schedule
- horizon curriculum
- best checkpoint
- generalization gap
- למה training objective אינו בר־השוואה ישירה ל־validation

אבל הן אינן עונות ישירות על שאלת המחקר:

> כמה היסטוריה נחוצה?

לכן בדוח של שבעה עמודים הן משניות.

#### המלצה

להעביר ל**נספח** את הגרסה הראשונה, עם שני panels.

היא עדיפה משום שהיא מפרידה בצורה ברורה בין:

- evaluation קבוע ובר־השוואה בחלק העליון
- training objective שמשתנה בגלל curriculum ו־teacher forcing בחלק התחתון

הגרסה השנייה צפופה יותר ומערבבת Loss ו־teacher forcing ratio באותו panel.

אם אין נספח או מגבלת המקום קשוחה, אפשר להשמיט את שתיהן ולהסתפק במשפט בסעיף 4.4:

> The best checkpoint was selected using full-horizon, free-running validation loss, while the training objective changed during curriculum learning and was therefore not used directly for checkpoint comparison.

הגרף לא נדרש כדי להוכיח את תוצאות הפרויקט.

---

### 8. גרף Straight מול Maneuver עבור כל המודלים

**הגרף:** Coastal suite — median FDE by motion type  
**קובץ:** `fig_straight_vs_maneuver.png`

הוא משווה:

- AR 9h / 12h / 18h / 24h
- Flat / Transformer / Adaptive / Sliding

#### למה הוא פחות מתאים?

הוא מכיל הרבה מידע, אבל לא מוסיף מסקנה מרכזית חדשה מעבר לגרף ה־AR בלבד.

הוא גם מערבב שתי שאלות:

1. איזה אורך היסטוריה טוב יותר?
2. איזו ארכיטקטורה טובה יותר?

כתוצאה מכך הוא פחות חד.

בנוסף, מופיעה אותה בעיית bucket:

> Bucket sizes differ slightly by history length.

ולכן חלק מהעמודות לא בהכרח מחושבות על קבוצות זהות.

גם התוצאה שבה Straight קשה בהרבה מ־Maneuver:

$$
\mathrm{Straight}\approx 39\text{–}63\,\mathrm{km}
\quad\text{לעומת}\quad
\mathrm{Maneuver}\approx 17\text{–}21\,\mathrm{km}
$$

דורשת הסבר משמעותי. היא יכולה לנבוע מכך שמסלולי Straight הם ספינות מהירות שעוברות מרחקים גדולים יותר, ולא בהכרח מכך שקל יותר לחזות Maneuver.

כלומר, ללא normalization לפי מרחק התנועה, הגרף עלול להטעות.

#### המלצה

**לא להכניס לגוף הדוח.**

אפשר:

- להעביר לנספח
- או להחליף אותו בגרף הדירוג הכללי שחסר בסעיף 6.1

אם בכל זאת שומרים אותו, צריך להוסיף גם normalized FDE או להבהיר:

> The higher straight-trajectory FDE may reflect longer traveled distances rather than lower directional predictability.

---

## מבנה מומלץ של הגרפים בדוח

### סעיף 4.4 — פונקציית הפסד ואימון

ללא גרף בגוף הדוח.

**בנספח:** Training history with curriculum and teacher forcing.

### סעיף 6.1 — דירוג כללי

איור חדש / משופר:

> Overall median FDE ranking

זה צריך להיות הגרף הראשון והחשוב ביותר.

לא להשתמש כאן בגרף Straight/Maneuver של כל המודלים.

### סעיף 6.2 — השפעת אורך ההיסטוריה

להכניס:

> Fixed-context AR LSTM — history length sweep

להסיר את נקודת ה־no-land או להעביר אותה ל־ablation.

### סעיף 6.3 — סוג תנועה

להכניס:

> AR LSTM — straight vs maneuver by history length

רצוי לחשב מחדש עם buckets קבועים ומשותפים.

### סעיף 6.4 — מודל אדפטיבי

להכניס:

> Adaptive multi-scale gate behavior

זה אחד הגרפים החשובים ביותר בדוח.

לאחריו להסביר בטקסט את Spearman ואת Random Forest.

### סעיף 7 — דיון

להכניס לכל היותר אחד:

- error growth לאורך 12 שעות, בתנאי שהוא מחושב על כל ה־test set
- או שלוש דוגמאות מסלול מייצגות

לא צריך את שניהם אם המקום מוגבל.

---

## סדר העדיפויות הסופי

### חובה

1. **Overall model ranking** — לשפר / לוודא (zoom על נוירוניים)
2. **History-length sweep** — בלי נקודת no-land
3. **Straight vs maneuver by AR history length** — לאחר תיקון buckets אם אפשר
4. **Adaptive gate behavior**

### טוב כתמיכה

5. **Error growth over horizon** — רק על כל ה־test set
6. **שלוש דוגמאות מסלול** — עדיף בנספח או בדיון

### נספח בלבד

7. **Training history** — הגרסה עם שני panels

### להסיר / לא לגוף הדוח

8. Training history duplicate with teacher forcing overlay  
9. All-model straight vs maneuver — אלא אם נותר מקום ויש buckets אחידים

---

## סיכום

כך כל גרף עושה עבודה ברורה:

| תפקיד | גרף |
|--------|------|
| דירוג | Overall model ranking |
| תשובה לשאלת ההיסטוריה | History-length sweep |
| הנעה להשערה האדפטיבית | Straight vs maneuver (AR) |
| למה ההשערה האדפטיבית לא הצליחה | Adaptive gate behavior |

---

## מיפוי קבצים קיימים

| המלצה | קובץ |
|--------|------|
| Ranking | `fig_model_ranking_fde.png` |
| History sweep | `fig_ar_context_sweep.png` |
| AR straight/maneuver | `fig_ar_straight_vs_maneuver.png` |
| Adaptive gate | `fig_adaptive_alphas.png` |
| Error vs horizon | `fig_error_vs_horizon.png` |
| Track examples | `fig_tracks_report_6panel.png` |
| Training (2 panels) | `fig_train_val_loss_ar12h.png` |
| Training + TF overlay | `fig_train_val_comparable_with_tf.png` |
| All-model straight/maneuver | `fig_straight_vs_maneuver.png` |

נתיב בסיס:

```text
data/results/USA Combined/unknown/exp_coastal/report_figures/
```
