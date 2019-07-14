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
                        
            IEnumerable<Sentimento> sentimentDatas = new List<Sentimento>
            {
                new Sentimento { SePositivo = true, Comentario = "muito bom" },
                new Sentimento { SePositivo = true, Comentario = "gostei disso" },
                new Sentimento { SePositivo = true, Comentario = "gostei do que vocês fizeram" },
                new Sentimento { SePositivo = true, Comentario = "bom trabalho" },
                new Sentimento { SePositivo = false, Comentario = "não gostei" },
                new Sentimento { SePositivo = false, Comentario = "muito ruim" }
            };

            MLContext mlContext = new MLContext(seed: 0);
            IDataView dataView = mlContext.Data.LoadFromEnumerable(sentimentDatas);
            TrainTestData splitDataView = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var estimator = mlContext.Transforms.Text.FeaturizeText
                (outputColumnName: "Features", inputColumnName: nameof(Sentimento.Comentario))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression
                (labelColumnName: "Label", featureColumnName: "Features"));

            var model = estimator.Fit(splitDataView.TrainSet);
            PredictionEngine<Sentimento, PredicacaoDeSentimento> predictionFunction 
                = mlContext.Model.CreatePredictionEngine<Sentimento, PredicacaoDeSentimento>(model);

            Sentimento sampleStatement = new Sentimento
            {
                Comentario = "tá bom"
            };
            var resultPrediction = predictionFunction.Predict(sampleStatement);

            Console.WriteLine(resultPrediction.Predicao);
            Console.ReadKey();          

        }
    }
}
