# Zubut Backend


## Conditionals
Short if
```js
if (true) console.log(false)
```
Long if.
```js
const part1 = true || false
const part2 = true || true
// This
if (part1 && part2 /* ...................... */) 
    console.log(true)
// Over this
if (part1 && part2 /* ...................... */) console.log(true)
```

Multiline conditional body
```js
// This (use braces)
if (part1 && part2 /* ...................... */) {
    console.log({
        "asd": "asd",
        "nan": "afgh",
        "asdf": "asdffdav"
    })
}
// Over this
if (part1 && part2 /* ...................... */)
    console.log({ "asd": "asd", "nan": "afgh", "asdf": "asdffdav" })
```

## Else / Else if 
```js
if (false) {
    console.log("ers")
} else {
    console.log("res")
}
```

## Promises

Use arrow functions on promise related functions.
```js
const b = ({a}) => (new Promise((resolve, reject) => {
    resolve(a)
}))

b({a: 5})
    .then(res => {
        /*
            ...
        */
        console.log(res)
    })
    .catch(err => console.log(err))
```

* Prefer Promise over callback
* Always return something
* Fail early

```js
// This
function myFunc (a) {
    if (a == null) return false
    return true
}
// Over this
function myFunc (a) {
    if (a != null) {
        let x = 2
        return x
    } else {
        return false
    }
}

console.log(myFunc(2))
```

## Module exported functions
```js
// Log errors to external service
// Chalk as a module
module.exports.exported = function () {
    return "I'm Here!"
}

function c () {
    console.log(module.exports.exported())
}

c()
```

## API / Rest

For a particular resource (_zUser as example_), set an attribute to a value in that instance use the id in the URL instead of in the body.

```
Instead of:  zUsers/setA   body: { userId: 1234, a: 2}
Do:          zUsers/1234/setA body: { a: 2 }
```
For definition of an endpoint, use resources instead of actions. As the action is implicit in the HTTP verb.

```
Instead of:  GET zUsers/getmyProfile
Do:          GET zUsers/MyProfile
```

Prefer findById over findOne when using id
```
Instead of:  findOne({where: {id: 123456432356}, include: ["asdf"], function (err, res) {})
Do:          findById(id, { include: ["asdf"] }, function (err, res) {})
```