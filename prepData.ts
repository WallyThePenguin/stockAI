import * as tf from '@tensorflow/tfjs-node'
import {CSVDataset, Dataset} from '@tensorflow/tfjs-data/dist/index'


interface DataType extends tf.TensorContainerObject {
    xs: tf.Tensor;
    ys: tf.Tensor;
}
export async function prepData(data: CSVDataset): Promise<tf.data.Dataset<tf.TensorContainer>>{
    //Since tensorflow is not made for ts, we have to optimize the data to not bug.
    
    const dataset = data as any as Dataset<DataType>;
    
    // Number of features is the number of column names minus one for the label
   // column.
   const numOfFeatures = (await data.columnNames()).length - 1;
      
   // Prepare the Dataset for training.
    const flattenedDataset =
    dataset
      .map(({xs, ys}) =>
        {
          // Convert xs(features) and ys(labels) from object form (keyed by
          // column name) to array form.
          return {xs:Object.values(xs), ys:Object.values(ys)};
        })
      .batch(10);
    return flattenedDataset
}