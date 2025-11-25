import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExternalLink } from "lucide-react";

const models = [
  {
    name: "Classification Pro",
    category: "Sınıflandırma",
    accuracy: "98.5%",
    description: "Görüntü ve metin sınıflandırma için optimize edilmiş derin öğrenme modeli.",
    tags: ["Deep Learning", "CNN", "Transfer Learning"],
  },
  {
    name: "Prediction Engine",
    category: "Regresyon",
    accuracy: "96.2%",
    description: "Zaman serisi analizi ve tahminleme için gelişmiş LSTM mimarisi.",
    tags: ["LSTM", "Time Series", "Forecasting"],
  },
  {
    name: "Cluster Analytics",
    category: "Kümeleme",
    accuracy: "94.8%",
    description: "Müşteri segmentasyonu ve anomali tespiti için unsupervised learning.",
    tags: ["K-Means", "DBSCAN", "Segmentation"],
  },
  {
    name: "NLP Analyzer",
    category: "Doğal Dil İşleme",
    accuracy: "97.1%",
    description: "Sentiment analizi ve metin madenciliği için transformer tabanlı model.",
    tags: ["BERT", "Transformers", "NLP"],
  },
  {
    name: "Vision Recognition",
    category: "Görüntü İşleme",
    accuracy: "99.1%",
    description: "Nesne tanıma ve görüntü segmentasyonu için state-of-the-art CNN.",
    tags: ["ResNet", "Object Detection", "Segmentation"],
  },
  {
    name: "Recommendation System",
    category: "Öneri Sistemi",
    accuracy: "95.5%",
    description: "Kişiselleştirilmiş öneriler için collaborative filtering algoritması.",
    tags: ["Collaborative", "Matrix Factorization", "Personalization"],
  },
];

const Models = () => {
  return (
    <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-b from-background to-card/30">
      <div className="container mx-auto">
        {/* Section Header */}
        <div className="text-center mb-16 max-w-3xl mx-auto">
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">
            Hazır <span className="text-primary">Modeller</span>
          </h2>
          <p className="text-lg text-muted-foreground">
            Çeşitli kullanım senaryoları için önceden eğitilmiş ve optimize edilmiş modeller
          </p>
        </div>

        {/* Models Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {models.map((model, index) => (
            <Card
              key={index}
              className="bg-card border-border hover:border-primary/50 transition-all duration-300 hover:shadow-glow group"
            >
              <CardHeader>
                <div className="flex items-start justify-between mb-2">
                  <Badge variant="secondary" className="bg-primary/10 text-primary border-primary/20">
                    {model.category}
                  </Badge>
                  <span className="text-sm font-semibold text-primary">{model.accuracy}</span>
                </div>
                <CardTitle className="text-xl group-hover:text-primary transition-colors">
                  {model.name}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4 text-sm">
                  {model.description}
                </p>
                <div className="flex flex-wrap gap-2 mb-4">
                  {model.tags.map((tag, tagIndex) => (
                    <Badge
                      key={tagIndex}
                      variant="outline"
                      className="text-xs border-border hover:border-primary/30"
                    >
                      {tag}
                    </Badge>
                  ))}
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full group-hover:border-primary group-hover:text-primary"
                >
                  Detayları Görüntüle
                  <ExternalLink className="ml-2 w-4 h-4" />
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Models;
