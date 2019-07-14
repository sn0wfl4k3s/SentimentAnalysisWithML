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
                new SentimentData { SePositivo = true, Sentimento = "muito bom" },
                new SentimentData { SePositivo = true, Sentimento = "gostei disso" },
                new SentimentData { SePositivo = true, Sentimento = "gostei do que vocês fizeram" },
                new SentimentData { SePositivo = true, Sentimento = "bom trabalho" },
                new SentimentData { SePositivo = false, Sentimento = "não gostei" },
                new SentimentData { SePositivo = false, Sentimento = "muito ruim" }
            };

            MLContext mlContext = new MLContext(seed: 0);
            IDataView dataView = mlContext.Data.LoadFromEnumerable<SentimentData>(sentimentDatas);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var estimator = mlContext.Transforms.Text.FeaturizeText
                (outputColumnName: "Features", inputColumnName: nameof(SentimentData.Sentimento))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression
                (labelColumnName: "Label", featureColumnName: "Features"));

            var model = estimator.Fit(splitDataView.TrainSet);
            PredictionEngine<SentimentData, SentimentPrediction> predictionFunction 
                = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            SentimentData sampleStatement = new SentimentData
            {
                Sentimento = "tá bom"
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);
            Console.WriteLine(resultPrediction.Predicao);
            Console.ReadKey();          

        }
    }
}
