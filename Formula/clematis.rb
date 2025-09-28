# typed: false
# frozen_string_literal: true
class Clematis < Formula
  include Language::Python::Virtualenv

  desc "Deterministic, offline-by-default multi-agent concept-graph engine"
  homepage "https://github.com/vecipher/Clematis3"
  # These three are substituted below after heredoc with env vars.
  url "https://github.com/vecipher/Clematis3/releases/download/v0.8.0a4/clematis-0.8.0a4.tar.gz"
  sha256 "c98b1b968f0b959ca2d1d58107d8b36a1525dcc197ae2bc3574e9eeb27c439a4"
  license "MIT"
  version "0.8.0a4"

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
