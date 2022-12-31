import * as tf from '@tensorflow/tfjs-node'
import {CSVDataset} from '@tensorflow/tfjs-data/dist/index'
export async function createDataset(csvPath): Promise<CSVDataset>{
    // We want to predict the next day of 2021, using 252 days worth of data.
    // We want to predict the Open value, and Close value of the next day:
    // So we make the label of "TIME" to true.
    const dataset = tf.data.csv(
        csvPath,{
            hasHeader: true,
            columnConfigs:{
            t: {
                isLabel: true
            }
            }
        }
    );
    return dataset
}