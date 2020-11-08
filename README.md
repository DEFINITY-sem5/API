The API is built on Flask.

Takes the input in the form of a URL, parses the article, takes Title & Text.

Runs the model on the parsed attributes, and gives the output. 

``` 
{
  "message": {
    "article_text": "legit news", 
    "article_title": "non-clickbait"
  }, 
  "type": "success"
}
```