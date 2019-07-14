using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    public class SentimentData
    {
        [LoadColumn(0)]
        public string Sentimento;

        [LoadColumn(1), ColumnName("Label")]
        public bool SePositivo;
    }

    public class SentimentPrediction : SentimentData
    {

        [ColumnName("PredictedLabel")]
        public bool Predicao { get; set; }

        public float Probabilidade { get; set; }

        public float Pontuacao { get; set; }
    }
}
