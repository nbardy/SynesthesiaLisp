## Magic Hat

A lisp dialogue for talking to images

Example:

```clojure
(let [people-count "How many people are in the image, answer zero if none?"
      person "What does the character look like in this image, describe their gender, expression, pose, position(left,right,center), hair color, clothing, and any key artistic details, Character Description:"
      non-person "What is the subject of this film scene?"
      people "Describe the main people in this film scene and their interaction, Description:"
      color-grade "Describe the color grade of this film scene in simple words, Description:"
      category "Is this scene focused around action, emotion, or artistic?, Choose One: "
      standard-camera "Is this a standard camera angle?"
      camera-detail (if (~= standard-camera "yes")
                      ""
                      "What is the camera angle? Describe the camera angle in simple words, Description:")
      is-color (? "Q: is color used in an interesting way? A:")
      color-detail (if (~= is-color "yes")
                     "Describe the intersting use of color in this scene, Description:"
                     "")]
  (let [subject (cond
                  (= people-count 0) non-person
                  (= people-count 1) person
                  (> people-count 1) people)
        details (let [more-details (case category
                                     "action" "What is the focus of this action scene?"
                                     "conversation" "What type of conversation is this?"
                                     "emotion" "What emotional content does the scene convey?"
                                     "artistic" "What is the artistic focus of the scene?")
                      result "Given the scene is a {subject} type scene
                              Given this question and answer:
                              {more-details:full}
                              
                              What is the best caption for this image?"]
                  result)
        person-count-string (cond
                              (= people-count 0) (str "No people")
                              (= people-count 1) (str "one person")
                              (> people-count 1) (str people-count (str " people")))
        total-caption (str "{person-count-string} {subject} details")
        critique-caption "Critique the caption: {total-caption}"]
    "Given the following:
     total caption:
     {total-caption}

     critique caption:
     {critique-caption}

     color grade:
     {color-grade}

     camera detail:
     {camera-detail}

     details:
     {details}

     color detail:
     {color-detail}

     camera detail:
     {camera-detail}

     And with the goal of generating a caption for this image.
     We want the caption to image mapping to be optimal for simple control.

     It should capture limited dense information about the image, but not required the user to write length descriptions
     to capture the information above. It should be simple to write and easy to understand.

     Now let's wite the final caption.

     caption:"))
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

