using Microsoft.ML.Data;

namespace SentimentAnalysis
{
    public class Sentimento
    {
        [LoadColumn(0)]
        public string Comentario;

        [LoadColumn(1), ColumnName("Label")]
        public bool SePositivo;
    }

    public class PredicacaoDeSentimento : Sentimento
    {

        [ColumnName("PredictedLabel")]
        public bool Predicao { get; set; }

        public float Probabilidade { get; set; }

        public float Pontuacao { get; set; }
    }
}
