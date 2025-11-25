import { Button } from "@/components/ui/button";
import { ArrowRight, Mail } from "lucide-react";

const CTA = () => {
  return (
    <section className="py-20 px-4 sm:px-6 lg:px-8 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute inset-0 bg-gradient-glow animate-glow-pulse pointer-events-none opacity-50" />
      
      <div className="container mx-auto relative z-10">
        <div className="max-w-4xl mx-auto text-center">
          {/* Content */}
          <h2 className="text-4xl sm:text-5xl font-bold mb-6">
            Projelerinizi <span className="text-primary">Bir Üst Seviyeye</span> Taşıyın
          </h2>
          <p className="text-lg text-muted-foreground mb-8 max-w-2xl mx-auto">
            Machine Learning ve Data Analysis çözümlerimiz ile verilerinizden maksimum değer çıkarın. Hemen başlayın ve farkı görün.
          </p>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Button 
              size="lg" 
              className="group bg-primary hover:bg-primary/90 text-primary-foreground shadow-glow transition-all duration-300 hover:scale-105"
            >
              <Mail className="mr-2 w-5 h-5" />
              İletişime Geçin
              <ArrowRight className="ml-2 w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Button>
            <Button 
              size="lg" 
              variant="outline"
              className="border-primary/30 hover:border-primary hover:bg-primary/10"
            >
              Ücretsiz Demo Talep Et
            </Button>
          </div>

          {/* Trust Indicators */}
          <div className="mt-12 pt-8 border-t border-border">
            <p className="text-sm text-muted-foreground mb-4">
              Güvenilir Ortaklar
            </p>
            <div className="flex flex-wrap items-center justify-center gap-8 opacity-50">
              <div className="text-2xl font-bold">TensorFlow</div>
              <div className="text-2xl font-bold">PyTorch</div>
              <div className="text-2xl font-bold">Scikit-learn</div>
              <div className="text-2xl font-bold">Keras</div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default CTA;
