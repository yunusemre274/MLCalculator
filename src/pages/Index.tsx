import Navigation from "@/components/Navigation";
import Hero from "@/components/Hero";
import Features from "@/components/Features";
import Models from "@/components/Models";
import CTA from "@/components/CTA";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      <main>
        <div id="home">
          <Hero />
        </div>
        <div id="features">
          <Features />
        </div>
        <div id="models">
          <Models />
        </div>
        <div id="contact">
          <CTA />
        </div>
      </main>
      <footer className="py-8 border-t border-border">
        <div className="container mx-auto px-4 text-center text-muted-foreground">
          <p>© 2024 ML Analytics. Tüm hakları saklıdır.</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
