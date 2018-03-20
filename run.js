var express = require('express');
var app = express();
var bodyParser = require('body-parser');
var fs=require('fs');


app.use(bodyParser.urlencoded({extended: true}));
app.use(bodyParser.json())
// Creates a server which runs on port 3000 and 
// can be accessed through localhost:3000
app.set('view engine', 'ejs');
 app.use(express.static(__dirname + '/public'));
 app.get('/', (req, res) => {
    //res.locals.session=req.session;
    res.render('index');
   // index=fs.readFileSync(__dirname+"/index.html");
    //res.send(index.toString());
 // res.sendFile('index', {root: __dirname });
});
// Function callName() is executed whenever 
// url is of the form localhost:3000/name
app.post('/predict', callName);
 
function callName(req, res) {
     
    // Use child_process.spawn method from 
    // child_process module and assign it
    // to variable spawn
    var spawn = require("child_process").spawn;
     console.log(req.body);
   
 //console.log(req.body.avgtemp,req.body.cloud,req.body.dr,req.body.grf,req.body.maxtemp,req.body.mintemp,req.body.pevap,req.body.wdf,req.body.rce,req.body.vp);    

    var process = spawn('python',["./hello.py",
                            req.body.avgtemp,
                            req.body.cloud,
                            req.body.dr,
                            req.body.grf,
                            req.body.maxtemp,
                            req.body.mintemp,
                            req.body.kevap,
                            req.body.wdf,
                            req.body.rce,
                            req.body.vp,
                            req.body.month,
                            req.body.place] );
 
    // Takes stdout data from script which executed
    // with arguments and send this data to res object
    process.stdout.on('data', function(data) {
        console.log('came from python');
        console.log(data.toString());
        res.send(data.toString());
    } )
}

app.listen(3000, function() {
    console.log('server running on port 3000');
} )