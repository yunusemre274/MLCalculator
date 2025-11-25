import { Card, CardContent } from "@/components/ui/card";
import { Database, TrendingUp, Zap, Shield, GitBranch, Cpu } from "lucide-react";

const features = [
  {
    icon: Database,
    title: "Veri İşleme",
    description: "Büyük veri setlerini hızlı ve etkili bir şekilde işleyin ve analiz edin.",
  },
  {
    icon: TrendingUp,
    title: "Tahminleme",
    description: "Gelişmiş ML algoritmaları ile gelecekteki trendleri tahmin edin.",
  },
  {
    icon: Zap,
    title: "Gerçek Zamanlı Analiz",
    description: "Verilerinizi anlık olarak işleyin ve sonuç alın.",
  },
  {
    icon: Shield,
    title: "Güvenli Altyapı",
    description: "Verileriniz end-to-end şifreleme ile korunur.",
  },
  {
    icon: GitBranch,
    title: "Model Versiyonlama",
    description: "Modellerinizi versiyonlayın ve en iyi performansı takip edin.",
  },
  {
    icon: Cpu,
    title: "Yüksek Performans",
    description: "GPU destekli işleme ile maksimum hız ve verimlilik.",
  },
];

const Features = () => {
  return (
    <section className="py-20 px-4 sm:px-6 lg:px-8 relative">
      <div className="container mx-auto">
        {/* Section Header */}
        <div className="text-center mb-16 max-w-3xl mx-auto">
          <h2 className="text-4xl sm:text-5xl font-bold mb-4">
            Güçlü <span className="text-primary">Özellikler</span>
          </h2>
          <p className="text-lg text-muted-foreground">
            Machine Learning ve Data Analysis ihtiyaçlarınız için her şey bir arada
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <Card
              key={index}
              className="bg-card border-border hover:border-primary/50 transition-all duration-300 hover:shadow-glow group cursor-pointer"
            >
              <CardContent className="p-6">
                <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                  <feature.icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-xl font-semibold mb-2 group-hover:text-primary transition-colors">
                  {feature.title}
                </h3>
                <p className="text-muted-foreground">
                  {feature.description}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
