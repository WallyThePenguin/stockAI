import * as tf from '@tensorflow/tfjs-node'
import {CSVDataset} from '@tensorflow/tfjs-data/dist/index'

export async function defineModel(data: CSVDataset, flattenedDataset: tf.data.Dataset<tf.TensorContainer>): Promise<tf.History>{
    // Number of features is the number of column names minus one for the label
   // column.
    const numOfFeatures = (await data.columnNames()).length - 1;
    
    // Define the model.
   const model = tf.sequential();
   model.add(tf.layers.dense({
     inputShape: [numOfFeatures],
     units: 1
   }));
   
   model.compile({
     optimizer: `sgd`,
     loss: 'meanSquaredError'
   });

   // Fit the model using the prepared Dataset
   return model.fitDataset(flattenedDataset, {
     epochs: 10,
     callbacks: {
       onEpochEnd: async (epoch, logs) => {
         console.log(epoch + ':' + logs.loss);
       }
     }
   });
}