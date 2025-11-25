import { Button } from "@/components/ui/button";
import { ArrowRight, Brain, BarChart3 } from "lucide-react";
import { useNavigate } from "react-router-dom";

const Hero = () => {
  const navigate = useNavigate();

  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Gradient Glow Background */}
      <div className="absolute inset-0 bg-gradient-glow animate-glow-pulse pointer-events-none" />
      
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="text-center max-w-4xl mx-auto animate-fade-in">
          {/* Icon badges */}
          <div className="flex items-center justify-center gap-4 mb-8">
            <div className="p-3 rounded-full bg-card border border-primary/20">
              <Brain className="w-6 h-6 text-primary" />
            </div>
            <div className="p-3 rounded-full bg-card border border-primary/20">
              <BarChart3 className="w-6 h-6 text-primary" />
            </div>
          </div>

          {/* Main heading */}
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold mb-6 bg-gradient-to-r from-foreground via-primary to-foreground bg-clip-text text-transparent">
            Machine Learning & Data Analysis
          </h1>
          
          {/* Subheading */}
          <p className="text-xl sm:text-2xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Verilerinizi güçlü makine öğrenmesi modelleri ile analiz edin. Gelişmiş yapay zeka çözümleri ile içgörüler keşfedin.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button 
              size="lg" 
              className="group bg-primary hover:bg-primary/90 text-primary-foreground shadow-glow transition-all duration-300 hover:scale-105"
              onClick={() => navigate("/dashboard")}
            >
              Başlayın
              <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Button>
            <Button 
              size="lg" 
              variant="outline"
              className="border-primary/30 hover:border-primary hover:bg-primary/10"
              onClick={() => navigate("/more-info")}
            >
              Daha Fazla Bilgi
            </Button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-8 mt-16 pt-8 border-t border-border">
            <div>
              <div className="text-3xl font-bold text-primary">99.9%</div>
              <div className="text-sm text-muted-foreground mt-1">Doğruluk Oranı</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-primary">50+</div>
              <div className="text-sm text-muted-foreground mt-1">Model Çeşidi</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-primary">24/7</div>
              <div className="text-sm text-muted-foreground mt-1">Analiz Desteği</div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent pointer-events-none" />
    </section>
  );
};

export default Hero;
