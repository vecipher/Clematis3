# typed: false
# frozen_string_literal: true
class Clematis < Formula
  include Language::Python::Virtualenv

  desc "Deterministic, offline-by-default multi-agent concept-graph engine"
  homepage "https://github.com/vecipher/Clematis3"
  # These three are substituted below after heredoc with env vars.
  url "https://github.com/vecipher/Clematis3/releases/download/v0.9.0a1/clematis-0.9.0a1.tar.gz"
  sha256 "6dd76b3fad0fb981b51821df4c1d4c82291c78f3f9ccfcf514a32cd1987bdf03"
  license "MIT"
  version "0.9.0a1"

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
