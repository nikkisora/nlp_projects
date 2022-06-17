var form = document.getElementById('textForm')
var text = document.getElementById('textArea')
var words = document.getElementById('words')


models = [
    'textrank',
    'topicrank',
    'positionrank',
    'summatextrank',
    'lexrank',
    'lsa',
    'luhn',
    'kl',
    'bart',
    'gpt2',
    't5',
    'xlnet'
]


form.onsubmit = function(e){
    e.preventDefault();

    models.forEach(model => {
        send(model)
    });



    e.preventDefault();
}


function send(model){
    (async () => {
        const response = await fetch('http://localhost:8080/summary',{
            method: 'POST',
            headers:{
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                'text': text.value,
                'max_words': words.value,
                'model': model
            }),
            mode: 'cors'
        })
        const summary = (await response.json()).summary
        console.log(summary)
        document.getElementById(model).innerHTML = summary
    })()
}