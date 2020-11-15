The API is built on Flask.

Takes the input in the form of a URL, parses the article, takes Title & Text.

Runs the model on the parsed attributes, and gives the output. 

``` 
{"message":
  {"art_text":"Delhi records 3,235 new Covid cases with 95 deaths in a day, government data shows. (File)\n\nDelhi recorded 3,235 new COVID-19 cases and 95 deaths due to the coronavirus disease in the last 24 hours, state government data shows.\n\nOn Diwali, infections in the national capital dropped to almost half of last week's average of 7,000 daily cases as just 21,098 tests were conducted - a third of the usually 60,000 tests conducted per day in Delhi since its second wave in September.\n\nThough the drop in testing was attributed to staff shortage because of the festival, it also led to speculation about what the actual number of new cases would have been as the positivity rate increased to 15.33 per cent.\n\nThe daily coronavirus figures, which are usually released around 10 pm every day, were shared just ahead of an urgent meeting with Home Minister Amit Shah today as COVID-19 spike continues unabated in the national capital. Delhi Chief Minister Arvind Kejriwal is likely to attend this meeting.\n\nSources said Mr Shah is concerned by the steady rise in Covid cases in Delhi, which is experiencing a third wave of infections taking the availability of ICU beds with ventilator support to an all-time low.\n\nIn view of rise in number of Covid patients with severe symptoms, the High Court had recently allowed the Delhi government to reserve 80 per cent beds in Intensive Care Units of 33 private hospitals for coronavirus cases.\n\nDelhi's daily Covid chart had started its upward climb 12 days ago. On November 3, the city reported a 24-hour surge of 6,725 cases after a few weeks of low numbers. On November 6, it crossed the 7,000 mark. A week later on November 11, it crossed the 8,000-mark, logging 8,593 cases - an all-time high for the city.\n\nAccording to an expert report released by the central government, Delhi is likely to see an average of 15,000 cases a day in the winter - double the current caseload.\n\nFor now, the total coronavirus case count in Delhi stands at 4,85,405, with 7,614 overall deaths and 39,990 active cases. More than 4.3 lakh people in the city have recovered from COVID-19 so far.",
  "art_title":"Delhi Covid Tests Down To Third On Diwali, New Cases Half The Average",
  "text_out":"legit news",
  "title_out":"non-clickbait"},
  "type":"success"}
```
