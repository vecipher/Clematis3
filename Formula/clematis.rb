# typed: false
# frozen_string_literal: true
class Clematis < Formula
  include Language::Python::Virtualenv

  desc "Deterministic, offline-by-default multi-agent concept-graph engine"
  homepage "https://github.com/vecipher/Clematis3"
  # These three are substituted below after heredoc with env vars.
  url "https://github.com/vecipher/Clematis3/releases/download/v0.10.3/clematis-0.10.3.tar.gz"
  sha256 "33bb8cb3faf29c1589f005e8a34be9ce0ff88258995538cf35f1503557e8c8ec"
  license "MIT"
  version "0.10.3"

  depends_on "python@3.13"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/clematis --version")
    system "#{bin}/clematis", "validate", "--json", "--help"
  end

  livecheck do
    url :stable
    strategy :github_latest
  end
end
