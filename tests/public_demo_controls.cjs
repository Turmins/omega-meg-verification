// Unit checks of the public demo controller with a small DOM fixture.
// This is not a visual browser test and does not use browser automation.
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const html=fs.readFileSync(path.join(__dirname,'../demo/index.html'),'utf8');
const data=JSON.parse(html.match(/<script id="public-aggregate-data" type="application\/json">(.*?)<\/script>/s)[1]);
const script=[...html.matchAll(/<script([^>]*)>(.*?)<\/script>/gs)].find(x=>!x[1].includes('application/json'))[2];
const context={module:{exports:{}}};vm.createContext(context);vm.runInContext(script,context);
const {decide,selectAggregate,initDemo}=context.module.exports;
class Element{
  constructor(value=''){this.value=value;this.options=[];this.textContent='';this.listeners={};}
  replaceChildren(){this.options=[];this.value='';}
  appendChild(option){this.options.push(option);if(this.options.length===1)this.value=option.value;}
  addEventListener(name,handler){this.listeners[name]=handler;}
  fire(name){assert.equal(typeof this.listeners[name],'function');this.listeners[name]();}
}
const nodes=new Map();
const doc={getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);},createElement:()=>new Element()};
doc.getElementById('dataset').value='v1';doc.getElementById('score').value='0.82';
doc.getElementById('threshold').value='0.60';doc.getElementById('truth').value='genuine';
initDemo(doc,data);
let aggregateChoices=0;
for(const [dataset,rows] of Object.entries(data.tables)){
  doc.getElementById('dataset').value=dataset;doc.getElementById('dataset').fire('change');
  for(const row of rows){
    for(const [id,value] of Object.entries({method:row.method,duration:row.duration_seconds,enrollment:row.enrollment_seconds||240,policy:row.policy||'static'}))doc.getElementById(id).value=String(value);
    doc.getElementById('method').fire('change');
    assert.equal(doc.getElementById('hter').textContent,(100*row.HTER).toFixed(2)+'%');
    assert.equal(doc.getElementById('auc').textContent,row.mean_fold_AUC===undefined?'—':row.mean_fold_AUC.toFixed(4));
    assert.ok(doc.getElementById('counts').textContent.includes(row.false_accepts+' / '+row.n_impostor));
    assert.equal(selectAggregate(data,{dataset,method:row.method,duration:row.duration_seconds,enrollment:row.enrollment_seconds||240,policy:row.policy||'static'}),row);
    aggregateChoices++;
  }
}
for(const [button,accepted,error] of [['genuine',true,null],['impostor',false,null],['false-accept',true,'FAR'],['false-reject',false,'FRR']]){
  doc.getElementById(button).fire('click');
  const result=decide(Number(doc.getElementById('score').value),Number(doc.getElementById('threshold').value),doc.getElementById('truth').value==='genuine');
  assert.equal(result.accepted,accepted);assert.equal(result.error,error);
  if(error)assert.ok(doc.getElementById('explanation').textContent.includes(error));
}
assert.equal(decide(.6,.6,true).accepted,true);
assert.throws(()=>decide(NaN,.6,true));
console.log(JSON.stringify({status:'passed',aggregate_choices:aggregateChoices,synthetic_buttons:4,equality_boundary:true,visual_browser_test:false}));
