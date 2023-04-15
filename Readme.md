## Magic Hat

A lisp dialogue for talking to multimodal LLMS

Example:

film.superprompt
```clojure
(let
 ^{:model "blip2"
   :model-args ["test_example.jpg"]}
 [people-count "How many people are in the image, answer zero if none?"
  subject (cond
            (= people-count 0) "What is in this image?"
            (= people-count 1) "Q: What does the person in this image look like?
                                 Notes: Add comments on the most interesting features, what is visualy stunning about this person?
                                 A:"
            (>= people-count 2) "Q: What does the person in this image look like?
                                 Notes: Add comments on the most interesting features, what is visualy stunning about this person?
                                 A:")
  total-caption (str "The Subject is ")
  critique-caption "Critique the caption: {subject}"]
  "Given the following:
    caption
    {total-caption}

    critique caption:
    {critique-caption}

    Generate a final caption densely that describes the most interesting parts of the image in the simplest way.")
~
```

Call in python

```
import magichat

result = magichat.magic("film.superprompt")
```




### Features:
LLM first, strings are by default sent to the server
Easy support for images with `let-image`
String Interpolation with new types:
	"GenResult" instead of string allows you to use either the llm result or gg
String interpolation with optional (:full)
It adopts a modern lisp syntax mostly inherited from Clojure

Image by default, no extra code required to chain your images around

Allows adapters:
Defaults 
Define your own python function to eval on your own model or cloud provider.


## Why?


This is mostly used as an embedded lang for small chains of prompts. A language designe for prompts first
with some helpers for chaining them. By using a new DSL we can introduce agressive defaults and do some simple
examples in a small amount of code.

These are meant to be small embedded prompt recipes in a bigger python program. Not a new programming language.

