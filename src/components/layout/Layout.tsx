import { ReactNode } from "react";
import Header from "./Header";
import DevModeBanner from "./DevModeBanner";

interface LayoutProps {
  children: ReactNode;
}

const Layout = ({ children }: LayoutProps) => {
  return (
    <div className="min-h-screen bg-background">
      <DevModeBanner />
      <Header />
      <main className="container px-4 py-8 md:px-8">{children}</main>
    </div>
  );
};

export default Layout;
