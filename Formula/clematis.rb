

# typed: false
# frozen_string_literal: true

class Clematis < Formula
  include Language::Python::Virtualenv

  desc "Deterministic, offline-by-default multi-agent concept-graph engine"
  homepage "https://github.com/vecipher/Clematis3"
  # CI will update url/sha256/version on tag publish
  url "https://github.com/vecipher/Clematis3/releases/download/v0.8.0a1/clematis-0.8.0a1.tar.gz"
  sha256 "TO_BE_FILLED_BY_CI"
  license "MIT"
  version "0.8.0a1"

  depends_on "python@3.11"

  def install
    virtualenv_install_with_resources
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/clematis --version")
    system "#{bin}/clematis", "validate", "--json", "--help"
  end
end