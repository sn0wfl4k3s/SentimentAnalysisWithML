using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using static Microsoft.ML.DataOperationsCatalog;

namespace SentimentAnalysis
{
    class Program
    {
        static void Main(string[] args)
        {
                        
            List<SentimentData> sentimentDatas = new List<SentimentData>
            {
                new SentimentData { Sentiment = true, SentimentText = "muito bom" },
                new SentimentData { Sentiment = true, SentimentText = "gostei disso" },
                new SentimentData { Sentiment = true, SentimentText = "gostei do que vocês fizeram" },
                new SentimentData { Sentiment = true, SentimentText = "bom trabalho" },
                new SentimentData { Sentiment = false, SentimentText = "não gostei" },
                new SentimentData { Sentiment = false, SentimentText = "muito ruim" }
            };

            MLContext mlContext = new MLContext(seed: 0);
            IDataView dataView = mlContext.Data.LoadFromEnumerable<SentimentData>(sentimentDatas);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var estimator = mlContext.Transforms.Text.FeaturizeText
                (outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression
                (labelColumnName: "Label", featureColumnName: "Features"));

            var model = estimator.Fit(splitDataView.TrainSet);
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction 
                = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            SentimentData sampleStatement = new SentimentData
            {
                SentimentText = "tá bom"
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine(resultPrediction.Prediction);
            Console.ReadKey();          

        }
    }
}
