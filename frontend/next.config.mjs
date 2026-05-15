// Nautilus proxy prefix — only applied when ASSET_PREFIX env var is set
const prefix = process.env.ASSET_PREFIX || '';

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactCompiler: true,

  output: 'standalone',
  turbopack: {
    root: process.cwd(),
  },

  assetPrefix: prefix || undefined,

  // This helps the proxy handle sub-routes
  trailingSlash: true,

  experimental: {
    serverActions: {
      bodySizeLimit: '100mb',
      allowedOrigins: (process.env.ALLOWED_SERVER_ACTION_ORIGINS || '').split(',').filter(Boolean),
    },
    proxyClientMaxBodySize: 104857600,
  },

  allowedDevOrigins: (process.env.ALLOWED_DEV_ORIGINS || '').split(',').filter(Boolean),

  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: [
              "default-src 'self'",
              "script-src 'self' 'unsafe-inline' 'unsafe-eval'",
              "style-src 'self' 'unsafe-inline'",
              "img-src 'self' data: blob:",
              "font-src 'self'",
              `connect-src 'self' ${process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000'}`,
              "frame-ancestors 'none'",
              "base-uri 'self'",
              "form-action 'self'",
            ].join('; '),
          },
          { key: 'X-Frame-Options', value: 'DENY' },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'Referrer-Policy', value: 'no-referrer' },
          { key: 'Permissions-Policy', value: 'camera=(), microphone=(), geolocation=()' },
          { key: 'X-XSS-Protection', value: '1; mode=block' },
        ],
      },
    ]
  },

  async rewrites() {
    const apiUrl = process.env.API_URL || 'http://localhost:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${apiUrl}/api/:path*`,
      },
      {
        source: '/health',
        destination: `${apiUrl}/health`,
      }
    ];
  },
};

export default nextConfig;
