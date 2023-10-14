package com.Fintech.OnlineBanking.config;

import java.io.IOException;
import org.springframework.lang.NonNull;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import com.Fintech.OnlineBanking.token.TokenRepository;
import io.jsonwebtoken.ExpiredJwtException;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@Component
public class JwtAuthenticationFilter extends OncePerRequestFilter {
	private final JwtService jwtService;
	private final UserDetailsService userDetailsService;
	private final TokenRepository tokenRepo;

	public JwtAuthenticationFilter(JwtService jwtService, UserDetailsService userDetailsService,
			TokenRepository tokenRepo) {
		super();
		this.jwtService = jwtService;
		this.userDetailsService = userDetailsService;
		this.tokenRepo = tokenRepo;
	}

	@Override
	protected void doFilterInternal(@NonNull HttpServletRequest request, @NonNull HttpServletResponse response,
			@NonNull FilterChain filterChain)

			throws ServletException, IOException {
		final String authHeader = request.getHeader("Authorization");
		final String jwt;
		final String userEmail;
		try {
			if (authHeader == null || !authHeader.startsWith("Bearer ")) {
				filterChain.doFilter(request, response);
				return;

			}
			jwt = authHeader.substring(7);
			userEmail = jwtService.extractUsername(jwt);
			if (userEmail != null && SecurityContextHolder.getContext().getAuthentication() == null) {
				UserDetails userDetails = this.userDetailsService.loadUserByUsername(userEmail);

				if (jwtService.isTokenValid(jwt, userDetails)) {
					UsernamePasswordAuthenticationToken authToken = new UsernamePasswordAuthenticationToken(userDetails,
							null, userDetails.getAuthorities());
					authToken.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
					SecurityContextHolder.getContext().setAuthentication(authToken);
				}
				filterChain.doFilter(request, response);
			}
		} catch (ExpiredJwtException e) {
			filterChain.doFilter(request, response);
			return;
		}
	}

}
