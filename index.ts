import * as tf from './node_modules/@tensorflow/tfjs-node';
import {createDataset} from './createData';
import {prepData} from './prepData';
import {defineModel} from './defineModel';

//Google Spreadsheet in a csv.
const url = `https://docs.google.com/spreadsheets/d/e/2PACX-1vTat_fWx4_1v66EfzzKrx7YgjbIhd8s6ROFIza3UKsc-48NGwu9Yk_A2E6wjkoW1b6pFklr50BcSmKS/pub?gid=360437723&single=true&output=csv`

console.log('Backend is:' + tf.getBackend());

(async () =>{ 
//So first, create the data from the google spreadsheet.
const dataSet = await createDataset(url)

//Then flatten and prep the data for a model.
const dataPrepped = await prepData(dataSet);

//Define the model.
const modelDefined = await defineModel(dataSet, dataPrepped);

//Print each individually to understand what is what.
//console.log(`Dataset` + dataSet.toArray() + `\n`);

//console.log(`Flattened Data:` + (await dataPrepped.toArray()).forEach( e=> console.log(e)), `\n`);

//console.log(`Model:` + [modelDefined.history].forEach(e => console.log(e)), `\n`);
})();