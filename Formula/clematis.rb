# typed: false
# frozen_string_literal: true
class Clematis < Formula
  include Language::Python::Virtualenv

  desc "Deterministic, offline-by-default multi-agent concept-graph engine"
  homepage "https://github.com/vecipher/Clematis3"
  # These three are substituted below after heredoc with env vars.
  url "https://github.com/vecipher/Clematis3/releases/download/v0.8.0a5/clematis-0.8.0a5.tar.gz"
  sha256 "c0f969ca48f3ee9f9e8ad18526dafae4c1a929b63c91dc9c09c346fb08b13d38"
  license "MIT"
  version "0.8.0a5"

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
