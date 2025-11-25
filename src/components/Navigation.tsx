import { Button } from "@/components/ui/button";
import { Brain, Menu } from "lucide-react";
import { useState } from "react";
import {
  Sheet,
  SheetContent,
  SheetTrigger,
} from "@/components/ui/sheet";

const Navigation = () => {
  const [isOpen, setIsOpen] = useState(false);

  const menuItems = [
    { label: "Anasayfa", href: "#home" },
    { label: "Özellikler", href: "#features" },
    { label: "Modeller", href: "#models" },
    { label: "İletişim", href: "#contact" },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-lg border-b border-border">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
              <Brain className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="text-xl font-bold">ML<span className="text-primary">Analytics</span></span>
          </div>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center gap-8">
            {menuItems.map((item) => (
              <a
                key={item.href}
                href={item.href}
                className="text-foreground hover:text-primary transition-colors"
              >
                {item.label}
              </a>
            ))}
          </div>

          {/* Desktop CTA */}
          <div className="hidden md:block">
            <Button className="bg-primary hover:bg-primary/90 text-primary-foreground">
              Başlayın
            </Button>
          </div>

          {/* Mobile Menu */}
          <Sheet open={isOpen} onOpenChange={setIsOpen}>
            <SheetTrigger asChild className="md:hidden">
              <Button variant="ghost" size="icon">
                <Menu className="w-6 h-6" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-[300px] sm:w-[400px]">
              <div className="flex flex-col gap-6 mt-8">
                {menuItems.map((item) => (
                  <a
                    key={item.href}
                    href={item.href}
                    className="text-lg font-medium hover:text-primary transition-colors"
                    onClick={() => setIsOpen(false)}
                  >
                    {item.label}
                  </a>
                ))}
                <Button className="bg-primary hover:bg-primary/90 text-primary-foreground w-full mt-4">
                  Başlayın
                </Button>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
